# frozen_string_literal: false
class Matrix
  class DiscreteAlgebraicRiccatiEquation
    EPSILON = Float::EPSILON
    SQRT_EPSILON = Math.sqrt(EPSILON)
    NEAR_UNIT_BAND = 1.0e-8
    ILL_CONDITIONED_PIVOT = 1.0e-6
    ILL_CONDITIONED_NUMBER = 1.0 / SQRT_EPSILON
    RESIDUAL_TOLERANCE = 1.0e-8
    STABILITY_MARGIN = 1.0e-6
    MAX_DOUBLING_ITERATIONS = 128
    MAX_NEWTON_ITERATIONS = 32

    Result = Struct.new(:success, :x, :k, :feedback_gain, :closed_loop, :reason,
                        :residual, :relative_residual, :symmetry_error,
                        :spectral_radius, :stability_margin,
                        :minimum_eigenvalue_x, :minimum_eigenvalue_s,
                        :condition_estimate, :stabilizability_pivot,
                        :stabilizability_radius, :doubling_iterations,
                        :newton_iterations, :regularization, :warnings,
                        :closed_loop_eigenvalues, :open_loop_eigenvalues,
                        :diagnostics) do
      def success?
        success
      end

      def to_a
        [x, k, diagnostics]
      end
      alias_method :to_ary, :to_a
    end

    def self.solve(a, b, q, r)
      new(a, b, q, r).solve
    end

    def initialize(a, b, q, r)
      @a = validate_matrix(a, :a)
      @b = validate_matrix(b, :b)
      @q = validate_matrix(q, :q)
      @r = validate_matrix(r, :r)
      validate_dimensions
      validate_weight_matrices
      @q = symmetrize(@q)
      @r = symmetrize(@r)

      @n = @a.row_count
      @open_eigenvalues = eigen_information(@a)
      @warnings = []
      @doubling_iterations = 0
      @newton_iterations = 0
      @regularization = 0.0
      @condition_estimate = 0.0
    end

    def solve
      maximum_weight = [
        1.0,
        frobenius_magnitude(@q),
        frobenius_magnitude(@r)
      ].max.to_f
      weight_scale = robust_sqrt(maximum_weight)
      scale_matrices(weight_scale)
      @result_weight_scale = weight_scale
      @result_weight_squared = robust_square(weight_scale)

      precheck = stabilizability_precheck
      return failed_result(precheck) unless precheck.empty?

      candidate = solve_candidate
      return failed_result(candidate) unless candidate.empty?

      validate_candidate
    end

    def validate_candidate
      x = symmetrize(@candidate)
      maximum_state_term = robust_square(frobenius_magnitude(@a)) * frobenius_magnitude(x)
      validation_scale = maximum_state_term.finite? ? 1.0 : robust_sqrt([
        1.0,
        frobenius_magnitude(@q),
        frobenius_magnitude(@r),
        robust_square(frobenius_magnitude(@b)) * frobenius_magnitude(x)
      ].max)
      squared_scale = robust_square(validation_scale)
      scaled_r = @r / squared_scale
      scaled_q = @q / squared_scale
      scaled_x = x / squared_scale
      s = scaled_r + @b.transpose * scaled_x * @b
      s = symmetrize(s)
      s_inverse = s.inverse
      k = s_inverse * @b.transpose * scaled_x * @a
      closed_loop = @a - @b * k
      atxa = @a.transpose * scaled_x * @a
      btxa = @b.transpose * scaled_x * @a
      rhs = scaled_q + atxa - @a.transpose * scaled_x * @b * s_inverse * btxa
      residual_matrix = scaled_x - rhs
      residual = frobenius_norm(residual_matrix) * squared_scale
      relative_residual = residual / (
        squared_scale *
          (1.0 + [frobenius_norm(scaled_x), frobenius_norm(scaled_q),
                  frobenius_norm(atxa)].max)
      )
      symmetry_error = frobenius_norm(scaled_x - scaled_x.transpose) /
                       (1.0 + frobenius_norm(scaled_x))

      x_information = eigen_information(scaled_x)
      closed_information = eigen_information(closed_loop)
      closed_values = closed_information.map { |item| item[:value] }
      spectral_radius = closed_values.map(&:abs).max
      eigenvalue_residual = closed_information.map { |item| item[:residual] }.max
      margin = closed_values.map { |value| 1.0 - value.abs }.min

      x_scale = squared_scale * (1.0 + frobenius_norm(scaled_x))
      minimum_x = x_information.map { |item| item[:value].real }.min *
                  squared_scale
      s_information = eigen_information(s)
      minimum_s = s_information.map { |item| item[:value].real }.min *
                  squared_scale
      @condition_estimate = [@condition_estimate, inverse_condition_estimate(s)].max

      warnings = []
      reason = nil
      candidate_asymmetry =
        frobenius_norm(scaled_x - scaled_x.transpose) /
        (1.0 + frobenius_norm(scaled_x))
      if minimum_x < -RESIDUAL_TOLERANCE * x_scale
        reason = :candidate_not_positive_semidefinite
      elsif candidate_asymmetry > RESIDUAL_TOLERANCE
        reason = :ill_conditioned_subspace
      elsif !relative_residual.finite?
        reason = :ill_conditioned_subspace
      elsif minimum_s <= EPSILON * squared_scale * (1.0 + frobenius_norm(s))
        reason = :ill_conditioned_subspace
      elsif symmetry_error > RESIDUAL_TOLERANCE
        reason = :residual_too_large
      elsif relative_residual > RESIDUAL_TOLERANCE
        reason = :residual_too_large
      elsif spectral_radius > 1.0 + STABILITY_MARGIN
        reason = :closed_loop_unstable
      elsif margin < STABILITY_MARGIN || eigenvalue_residual >= margin
        reason = :spectrum_near_unit_circle
      end

      warnings << :closed_loop_near_unit_circle if reason.nil? && margin < NEAR_UNIT_BAND
      warnings << :large_condition_estimate if reason.nil? &&
        @condition_estimate > ILL_CONDITIONED_NUMBER

      output_x = validation_scale == 1.0 ? x * @result_weight_squared : x
      output_k = k
      build_result(reason.nil?, output_x, output_k, closed_loop, reason,
                   residual: residual,
                   relative_residual: relative_residual,
                   symmetry_error: symmetry_error,
                   spectral_radius: spectral_radius,
                   stability_margin: margin,
                   minimum_eigenvalue_x: minimum_x,
                   minimum_eigenvalue_s: minimum_s,
                   condition_estimate: @condition_estimate,
                   eigenvalue_residual: eigenvalue_residual,
                   closed_loop_eigenvalues: closed_values,
                   warnings: @warnings + warnings)
    rescue ExceptionForMatrix::ErrNotRegular, ZeroDivisionError
      build_result(false, nil, nil, nil, :ill_conditioned_subspace)
    end

    def scale_matrices(weight_scale)
      @q = @q / robust_square(weight_scale)
      @r = @r / robust_square(weight_scale)
      @q = symmetrize(@q)
      @r = symmetrize(@r)
    end

    def robust_sqrt(value)
      Math.exp(Math.log(value) / 2.0)
    end

    def robust_square(value)
      Math.exp(Math.log(value) * 2.0)
    end

    def failed_result(information)
      build_result(false, nil, nil, nil, information[:reason],
                   stabilizability_pivot: information[:stabilizability_pivot],
                   stabilizability_radius: information[:stabilizability_radius])
    end

    def build_result(success, x, k, closed_loop, reason, values = {})
      diagnostics = {
        residual: values[:residual],
        relative_residual: values[:relative_residual],
        symmetry_error: values[:symmetry_error],
        spectral_radius: values[:spectral_radius],
        stability_margin: values[:stability_margin],
        eigenvalue_residual: values[:eigenvalue_residual],
        minimum_eigenvalue_x: values[:minimum_eigenvalue_x],
        minimum_eigenvalue_s: values[:minimum_eigenvalue_s],
        condition_estimate: @condition_estimate.nonzero? ? @condition_estimate : nil,
        stabilizability_pivot: values[:stabilizability_pivot] || @stabilizability_pivot,
        stabilizability_radius: values[:stabilizability_radius] || @stabilizability_radius,
        doubling_iterations: @doubling_iterations.zero? ? nil : @doubling_iterations,
        newton_iterations: @newton_iterations.zero? ? nil : @newton_iterations,
        regularization: @regularization.zero? ? nil : @regularization,
        warnings: values[:warnings] || @warnings,
        open_loop_eigenvalues: @open_eigenvalues.map { |item| item[:value] },
        closed_loop_eigenvalues: values[:closed_loop_eigenvalues] || []
      }.freeze

      Result.new(success, x, k, k, closed_loop, reason,
                 diagnostics[:residual],
                 diagnostics[:relative_residual],
                 diagnostics[:symmetry_error],
                 diagnostics[:spectral_radius],
                 diagnostics[:stability_margin],
                 diagnostics[:minimum_eigenvalue_x],
                 diagnostics[:minimum_eigenvalue_s],
                 diagnostics[:condition_estimate],
                 diagnostics[:stabilizability_pivot],
                 diagnostics[:stabilizability_radius],
                 diagnostics[:doubling_iterations],
                 diagnostics[:newton_iterations],
                 diagnostics[:regularization],
                 diagnostics[:warnings],
                 diagnostics[:closed_loop_eigenvalues],
                 diagnostics[:open_loop_eigenvalues],
                 diagnostics)
    end

    private

    def validate_matrix(matrix, name)
      raise TypeError, "#{name} must be a Matrix" unless matrix.is_a?(Matrix)
      raise ExceptionForMatrix::ErrDimensionMismatch unless matrix.square? || name == :b
      converted = matrix.map(&:to_f)
      unless converted.real? && converted.all? { |entry| entry.finite? }
        raise ArgumentError, "#{name} must contain finite real numbers"
      end
      converted
    end

    def validate_dimensions
      if @a.empty? || @b.empty? || @q.empty? || @r.empty?
        raise ExceptionForMatrix::ErrDimensionMismatch, "matrices must not be empty"
      end
      unless @a.square? && @q.square? && @r.square?
        raise ExceptionForMatrix::ErrDimensionMismatch, "A, Q and R must be square"
      end
      unless @q.row_count == @a.row_count
        raise ExceptionForMatrix::ErrDimensionMismatch,
              "Q must have the same dimension as A"
      end
      unless @b.row_count == @a.row_count
        raise ExceptionForMatrix::ErrDimensionMismatch,
              "B must have the same number of rows as A"
      end
      unless @r.row_count == @b.column_count
        raise ExceptionForMatrix::ErrDimensionMismatch,
              "R must have the same dimension as B's columns"
      end
    end

    def validate_weight_matrices
      unless symmetric?(@q, SQRT_EPSILON * (1.0 + frobenius_magnitude(@q)))
        raise ArgumentError, "Q must be symmetric positive semidefinite"
      end
      unless symmetric?(@r, SQRT_EPSILON * (1.0 + frobenius_magnitude(@r)))
        raise ArgumentError, "R must be symmetric positive definite"
      end

      q_tolerance = SQRT_EPSILON * (1.0 + frobenius_magnitude(@q))
      unless eigen_information(@q).all? { |item| item[:value] >= -q_tolerance }
        raise ArgumentError, "Q must be positive semidefinite"
      end

      r_values = eigen_information(@r).map { |item| item[:value] }
      unless r_values.all? { |value| value > 0.0 }
        raise ArgumentError, "R must be positive definite"
      end
    end

    def symmetric?(matrix, tolerance)
      frobenius_norm(matrix - matrix.transpose) <= tolerance
    end

    def frobenius_norm(matrix)
      Math.sqrt(matrix.each.map { |entry| entry.abs2 }.inject(0.0, &:+))
    end

    def frobenius_magnitude(matrix)
      matrix.each.map(&:abs).max.to_f
    end

    def eigen_information(matrix)
      decomposition = matrix.eigensystem
      matrix_scale = 1.0 + frobenius_norm(matrix)
      decomposition.eigenvalues.zip(decomposition.eigenvectors).map do |value, vector|
        vector_norm = vector.magnitude
        residual = (matrix * vector - value * vector).magnitude
        residual /= vector_norm.nonzero? || 1.0
        { value: value, residual: residual / matrix_scale }
      end
    rescue StandardError
      [{ value: Complex(Float::NAN), residual: Float::INFINITY }] * matrix.row_count
    end

    def stabilizability_precheck
      minimum_pivot = Float::INFINITY
      critical_radius = nil

      @open_eigenvalues.each do |information|
        value = information[:value]
        uncertainty = information[:residual]
        radius = value.abs + uncertainty
        next if radius < 1.0 - NEAR_UNIT_BAND

        pbh = Matrix.hstack(value * Matrix.identity(@a.row_count) - @a, @b)
        rank_information = qr_rank_with_column_pivoting(pbh)
        minimum_pivot = [minimum_pivot, rank_information[:minimum_pivot]].min
        critical_radius = [critical_radius || radius, radius].max
        scale = rank_information[:scale]
        deficient = rank_information[:rank] < @a.row_count
        nearly_deficient = rank_information[:minimum_pivot] < ILL_CONDITIONED_PIVOT * scale

        if deficient || nearly_deficient
          reason = if (1.0 - NEAR_UNIT_BAND..1.0 + NEAR_UNIT_BAND).cover?(radius) &&
                      (!deficient || nearly_deficient)
            :spectrum_near_unit_circle
          elsif deficient
            :not_stabilizable
          else
            :ill_conditioned_subspace
          end
          return failure_information(reason, minimum_pivot, critical_radius)
        end
      end

      @stabilizability_pivot = minimum_pivot == Float::INFINITY ? nil : minimum_pivot
      @stabilizability_radius = critical_radius
      {}
    end

    def qr_rank_with_column_pivoting(matrix)
      rows = matrix.to_a.map(&:dup)
      row_count = matrix.row_count
      column_count = matrix.column_count
      pivot_count = [row_count, column_count].min
      column_norms = Array.new(column_count) do |column|
        Math.sqrt(row_count.times.inject(0.0) do |sum, row|
          sum + rows[row][column].abs2
        end)
      end
      scale = [column_norms.max, 1.0].max
      rank = 0
      minimum_pivot = Float::INFINITY

      pivot_count.times do |k|
        pivot_column = (k...column_count).max_by { |column| column_norms[column] }
        if pivot_column != k
          rows.each { |row| row[k], row[pivot_column] = row[pivot_column], row[k] }
          column_norms[k], column_norms[pivot_column] =
            column_norms[pivot_column], column_norms[k]
        end

        pivot_abs = column_norms[k]
        minimum_pivot = [minimum_pivot, pivot_abs].min
        break if pivot_abs <= EPSILON * scale
        rank += 1
        next if k == row_count - 1

        vector = Array.new(row_count - k) { |row| Complex(rows[k + row][k]) }
        vector_norm = pivot_abs
        phase = vector.first.zero? ? Complex(1.0) : vector.first / vector.first.abs
        vector[0] -= -vector_norm * phase
        householder_norm = vector.inject(0.0) { |sum, entry| sum + entry.abs2 }
        tau = householder_norm.zero? ? 0.0 : 2.0 / householder_norm

        (k + 1...column_count).each do |column|
          dot = (k...row_count).inject(Complex(0.0)) do |sum, row|
            sum + vector[row - k].conj * rows[row][column]
          end
          (k...row_count).each do |row|
            rows[row][column] -= tau * vector[row - k] * dot
          end
        end

        rows[k][k] = -vector_norm * phase
        (k + 1...row_count).each { |row| rows[row][k] = Complex(0.0) }
        (k + 1...column_count).each do |column|
          column_norms[column] = Math.sqrt(((k + 1)...row_count).inject(0.0) do |sum, row|
            sum + rows[row][column].abs2
          end)
        end
      end

      { rank: rank, minimum_pivot: minimum_pivot, scale: scale }
    end

    def failure_information(reason, pivot, radius)
      {
        reason: reason,
        stabilizability_pivot: pivot == Float::INFINITY ? nil : pivot,
        stabilizability_radius: radius
      }
    end

    def solve_candidate
      result = doubling(@q)
      return {} if result.empty?

      regularized = regularized_doubling
      return regularized if regularized

      result
    end

    def doubling(source_q)
      a = @a
      g = source_q
      h = @b * @r.inverse * @b.transpose
      a_scale = 1.0 + frobenius_norm(@a)
      MAX_DOUBLING_ITERATIONS.times do |iteration|
        old_a = a
        gh = g * h
        hg = h * g
        if_gh = Matrix.identity(@n) + gh
        if_hg = Matrix.identity(@n) + hg
        @condition_estimate = [@condition_estimate, inverse_condition_estimate(if_gh)].max
        inverse_gh = if_gh.inverse
        inverse_hg = if_hg.inverse

        a = old_a * inverse_hg * old_a
        g = g + old_a.transpose * inverse_gh * g * old_a
        h = h + old_a * inverse_hg * h * old_a.transpose
        @doubling_iterations = iteration + 1

        unless finite_matrix?(a) && finite_matrix?(g) && finite_matrix?(h)
          return { reason: :ill_conditioned_subspace }
        end

        a_norm = frobenius_norm(a)
        relative_a = a_norm / a_scale
        if relative_a <= 1.0e-14
          @candidate = g
          return {}
        end
      end

      { reason: :iteration_did_not_converge }
    rescue ExceptionForMatrix::ErrNotRegular, ZeroDivisionError
      { reason: :ill_conditioned_subspace }
    end

    def regularized_doubling
      scale = 1.0 + frobenius_norm(@a)
      @regularization = 1.0e-10 * scale * scale
      result = doubling(@q + Matrix.scalar(@n, @regularization))
      return result unless result.empty?

      newton_refine
    end

    def newton_refine
      x = @candidate
      previous_change = Float::INFINITY

      MAX_NEWTON_ITERATIONS.times do
        s = @r + @b.transpose * x * @b
        @condition_estimate = [@condition_estimate, inverse_condition_estimate(s)].max
        k = s.inverse * @b.transpose * x * @a
        closed_loop = @a - @b * k
        rhs = symmetrize(@q + k.transpose * @r * k)
        updated = solve_stein(closed_loop, rhs)
        return { reason: :ill_conditioned_subspace } unless updated

        updated = symmetrize(updated)
        change = frobenius_norm(updated - x) / (1.0 + frobenius_norm(updated))
        x = updated
        @newton_iterations += 1
        if change < 1.0e-13
          @candidate = x
          return {}
        end
        if change > previous_change
          return { reason: :iteration_did_not_converge }
        end
        previous_change = change
      end

      { reason: :iteration_did_not_converge }
    rescue ExceptionForMatrix::ErrNotRegular, ZeroDivisionError
      { reason: :ill_conditioned_subspace }
    end

    def solve_stein(closed_loop, rhs)
      n = @n
      operator = Array.new(n * n) { Array.new(n * n, 0.0) }
      n.times do |input_row|
        n.times do |input_column|
          row = input_row * n + input_column
          operator[row][row] = 1.0
          n.times do |output_row|
            n.times do |output_column|
              operator[row][output_row * n + output_column] -=
                closed_loop[output_row, input_row] * closed_loop[output_column, input_column]
            end
          end
        end
      end

      vector = []
      n.times do |column|
        n.times { |row| vector << rhs[row, column] }
      end
      solution = Matrix.send(:new, operator, operator.size).lup.solve(Vector.elements(vector, false))
      Matrix.build(n) { |row, column| solution[row * n + column] }
    rescue ExceptionForMatrix::ErrNotRegular, ZeroDivisionError
      nil
    end

    def inverse_condition_estimate(matrix)
      inverse = matrix.inverse
      estimated = frobenius_norm(matrix) * frobenius_norm(inverse) / @n
      estimated.nan? || estimated.infinite? ? Float::INFINITY : estimated
    rescue ExceptionForMatrix::ErrNotRegular, ZeroDivisionError
      Float::INFINITY
    end

    def symmetrize(matrix)
      (matrix + matrix.transpose) / 2.0
    end

    def finite_matrix?(matrix)
      matrix.all? { |entry| entry.finite? }
    end
  end
end
