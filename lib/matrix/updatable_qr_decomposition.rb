# frozen_string_literal: false

class Matrix
  # Maintains a column-pivoted QR factorization, AP = QR, of a changing row
  # window.  Additions use a Givens rotation and deletions use a row-coordinate
  # rotation followed by a small rank-revealing factorization.  The current
  # rows and right hand sides are retained, but removed history is not kept.
  # A full refactorization occurs for structural rank loss at the square
  # boundary, a failed deletion update, or a periodic orthogonality check.
  class UpdatableQRDecomposition
    include Matrix::ConversionHelper
    include ExceptionForMatrix

    EPSILON = Float::EPSILON
    ORTHOGONALITY_PERIOD = 32
    ORTHOGONALITY_LIMIT = Math.sqrt(EPSILON)
    RANK_LOSS_LIMIT = Math.sqrt(EPSILON)

    # Holds the selected solution, least-squares residual norm, and the
    # rank/consistency status explaining which least-squares convention was used.
    Result = Struct.new(:solution, :residual_norm, :solvable,
                        :rank_deficient, :solution_convention,
                        :residual_norms, :solvability_tolerance,
                        keyword_init: true)

    attr_reader :column_count, :rank, :pivots, :update_count,
                :rebuild_count, :last_rebuild_reason, :rank_tolerance_option,
                :orthogonality_period, :orthogonality_limit,
                :rank_loss_limit

    def initialize(matrix, rhs = nil, rank_tolerance: nil,
                   orthogonality_period: ORTHOGONALITY_PERIOD,
                   orthogonality_limit: ORTHOGONALITY_LIMIT,
                   rank_loss_limit: RANK_LOSS_LIMIT)
      raise TypeError, "Expected Matrix but got #{matrix.class}" unless matrix.is_a?(Matrix)
      if rank_tolerance && (!rank_tolerance.is_a?(Numeric) || rank_tolerance < 0)
        raise ArgumentError, "rank tolerance must be a non-negative number"
      end

      @column_count = matrix.column_count
      @rows = finite_matrix(matrix).map(&:dup)
      @rhs_columns = rhs.is_a?(Matrix) ? rhs.column_count : 0
      @rhs_vector = false
      @rhs = initialize_rhs(rhs, @rows.size)
      @pivots = (0...@column_count).to_a
      @rank = 0
      @q = Array.new(@rows.size) { [] }
      @r = []
      @frobenius_squared = 0.0
      @update_count = 0
      @rebuild_count = 0
      @last_rebuild_reason = nil
      @rank_tolerance_option = rank_tolerance&.to_f
      @orthogonality_period = orthogonality_period
      @orthogonality_limit = orthogonality_limit
      @rank_loss_limit = rank_loss_limit
      rebuild!("initial factorization")
      @rebuild_count = 0
      @last_rebuild_reason = nil
    end

    def row_count
      @rows.size
    end

    def q
      Matrix.send(:new, deep_copy(@q), @rank)
    end

    def r
      Matrix.send(:new, deep_copy(@r), @column_count)
    end

    def rhs
      raise ArgumentError, "no right hand side is stored in this factorization" if @rhs_columns.zero?
      Matrix.send(:new, deep_copy(@rhs), @rhs_columns)
    end

    def rank_tolerance
      return @rank_tolerance_option if @rank_tolerance_option
      return 0.0 if @frobenius_squared.zero?
      EPSILON * [row_count, @column_count, 1].max * Math.sqrt(@frobenius_squared)
    end

    def rank_deficient?
      @rank < @column_count
    end

    def empty?
      @rows.empty?
    end

    def add_row(row, rhs = nil)
      append_row!(row, rhs)
      self
    end
    alias_method :append_row, :add_row

    def add_rows(rows, rhs = nil)
      converted = convert_rows(rows)
      converted_rhs = convert_rhs_rows(rhs, converted.size)
      converted.each_with_index { |row, i| append_row!(row, converted_rhs[i]) }
      self
    end
    alias_method :append_rows, :add_rows

    def delete_row(index = -1)
      remove_row!(index)
      self
    end
    alias_method :remove_row, :delete_row

    def replace_row(index, row, rhs = nil)
      normalized_index = Matrix::CoercionHelper.check_int(index, @rows.size, :index)
      snapshot = state_snapshot
      new_row = finite_row(row, @column_count)
      new_rhs = finite_rhs_row(rhs)

      begin
        removed_row = @rows.delete_at(normalized_index)
        @rhs.delete_at(normalized_index) if @rhs_columns.nonzero?
        @frobenius_squared -= row_norm(removed_row) ** 2
        incremental_delete!(normalized_index)

        @rows << new_row
        @rhs << new_rhs if @rhs_columns.nonzero?
        incremental_append!(new_row)
        move_last_row_to!(normalized_index)
      rescue NumericalRebuild
        restore_state(snapshot)
        replacement_rows = snapshot[:rows].map(&:dup)
        replacement_rhs = snapshot[:rhs].map(&:dup)
        replacement_rows[normalized_index] = new_row
        replacement_rhs[normalized_index] = new_rhs if @rhs_columns.nonzero?
        @rows = replacement_rows
        @rhs = replacement_rhs
        rebuild!("numerical degradation while replacing row #{normalized_index}")
      end

      @update_count += 1
      periodic_orthogonality_check!
      self
    end
    alias_method :update_row, :replace_row

    def rebuild!(reason = "manual rebuild")
      q, r, pivots, rank, frobenius_squared =
        factorize_rows(@rows, configured_rank_tolerance(@rows))
      @q = q
      @r = r
      @pivots = pivots
      @rank = rank
      @frobenius_squared = frobenius_squared
      @rebuild_count += 1
      @last_rebuild_reason = reason
      self
    end

    def solve(right_hand_sides = :not_given)
      if right_hand_sides.equal?(:not_given)
        raise ArgumentError, "no right hand side is stored in this factorization" if @rhs_columns.zero?
        b = deep_copy(@rhs)
        return_vector = @rhs_vector
      else
        b = rhs_argument(right_hand_sides, @rows.size)
        return_vector = flat_rhs?(right_hand_sides)
      end

      projected = transpose_multiply(@q, b)
      solution = minimum_norm_solution(projected)
      residual = matrix_subtract(b, matrix_multiply(rows_in_original_order, solution))
      norms = column_norms(residual)
      norm = Math.sqrt(norms.sum(0.0) { |value| value.abs2 })
      scale = [matrix_scale, frobenius_norm(b), 1.0].max
      tolerance = EPSILON * [row_count, @column_count, b[0]&.size || 0, 1].max * scale
      convention = rank_deficient? ?
        "minimum-norm least squares solution" :
        "unique least squares solution"
      answer = return_vector ? column_to_vector(solution, 0) :
        Matrix.send(:new, solution, b[0]&.size || 0)
      residual_answer = return_vector ? norms.first :
        Matrix.send(:new, norms.map { |value| [value] }, 1)

      Result.new(solution: answer, residual_norm: norm,
                 solvable: norms.all? { |value| value <= tolerance },
                 rank_deficient: rank_deficient?,
                 solution_convention: convention,
                 residual_norms: residual_answer,
                 solvability_tolerance: tolerance)
    end

    private

    class NumericalRebuild < StandardError; end
    private_constant :NumericalRebuild

    def initialize_rhs(value, size)
      return Array.new(size) { [] } if value.nil?
      @rhs_vector = flat_rhs?(value)
      rhs_argument(value, size)
    end

    def flat_rhs?(value)
      value.is_a?(Vector) || value.is_a?(Numeric) ||
        (value.is_a?(Array) && (value.empty? || !value.first.is_a?(Array)))
    end

    def rhs_argument(value, size)
      rows = case value
      when Matrix
        finite_matrix(value)
      when Vector
        vector = value.to_a
        finite_values(vector, "right hand side")
        vector.map { |entry| [entry.to_f] }
      when Numeric
        finite_values([value], "right hand side")
        [[value.to_f]]
      when Array
        converted = value.map(&:dup)
        if converted.empty? || !converted.first.is_a?(Array)
          finite_values(converted, "right hand side")
          converted.map { |entry| [entry.to_f] }
        else
          finite_rows(converted, nil, "right hand side")
          converted
        end
      else
        raise TypeError, "right hand side must be a Matrix, Vector, or Array"
      end

      raise ErrDimensionMismatch,
        "right hand side has #{rows.size} rows, expected #{size}" if rows.size != size
      columns = rows.first&.size || (size.zero? ? (@rhs_columns.nonzero? ? @rhs_columns : 0) : 0)
      columns = 1 if columns.zero? && size.zero? && value.is_a?(Vector)
      if @rhs_columns.nonzero? && columns != @rhs_columns
        raise ErrDimensionMismatch,
          "right hand side has #{columns} columns, expected #{@rhs_columns}"
      end
      @rhs_columns = columns
      rows
    end

    def finite_matrix(matrix)
      rows = matrix.to_a
      finite_rows(rows, matrix.column_count, "matrix")
      rows
    end

    def finite_rows(rows, expected_columns, name)
      rows.each_with_index do |row, row_index|
        unless row.is_a?(Array)
          raise TypeError, "#{name} row #{row_index} must be an Array"
        end
        if expected_columns && row.size != expected_columns
          raise ErrDimensionMismatch,
            "#{name} row #{row_index} has #{row.size} columns, expected #{expected_columns}"
        end
        finite_values(row, name)
        row.map!(&:to_f)
      end
    end

    def finite_values(values, name)
      values.each do |value|
        unless value.is_a?(Numeric) && value.real? && value.finite?
          raise ArgumentError, "all #{name} values must be finite real numbers"
        end
      end
    end

    def finite_row(row, expected_columns)
      values = row.is_a?(Array) ? row.dup : convert_to_array(row)
      raise ErrDimensionMismatch,
        "row has #{values.size} columns, expected #{expected_columns}" unless values.size == expected_columns
      finite_values(values, "row")
      values.map(&:to_f)
    end

    def finite_rhs_row(value)
      return [] if @rhs_columns.zero? && (value.nil? || @rows.empty?)
      raise ArgumentError, "a right hand side row is required" if value.nil?
      values = if value.is_a?(Array)
        value.first.is_a?(Array) ? value.first : value
      else
        converted = convert_to_array(value)
        converted.is_a?(Array) && converted.first.is_a?(Array) ? converted.first : converted
      end
      if @rhs_columns.zero?
        raise ArgumentError,
          "cannot attach right hand sides after rows were added without them" unless @rows.empty?
        @rhs_columns = values.size
      end
      raise ErrDimensionMismatch,
        "right hand side has #{values.size} columns, expected #{@rhs_columns}" unless values.size == @rhs_columns
      finite_values(values, "right hand side")
      values.map(&:to_f)
    end

    def convert_rows(rows)
      converted = case rows
      when Matrix
        finite_matrix(rows)
      when Array
        rows.first.is_a?(Array) ? rows.map(&:dup) : [rows.dup]
      else
        [convert_to_array(rows)]
      end
      converted.each { |row| finite_values(row, "row") }
      converted.map { |row| row.map(&:to_f) }
    end

    def convert_rhs_rows(value, count)
      return Array.new(count) { [] } if @rhs_columns.zero? && value.nil?
      if @rhs_columns.zero?
        raise ArgumentError,
          "cannot attach right hand sides after rows were added without them" unless @rows.empty?
        return rhs_argument(value, count)
      end
      matrix = value.is_a?(Matrix) ? finite_matrix(value) : rhs_argument(value, count)
      raise ErrDimensionMismatch,
        "right hand side has #{matrix.size} rows, expected #{count}" unless matrix.size == count
      matrix
    end

    def append_row!(row, rhs_value)
      new_row = finite_row(row, @column_count)
      new_rhs = finite_rhs_row(rhs_value)
      snapshot = state_snapshot

      begin
        @rows << new_row
        @rhs << new_rhs if @rhs_columns.nonzero?
        incremental_append!(new_row)
      rescue NumericalRebuild
        restore_state(snapshot)
        @rows << new_row
        @rhs << new_rhs if @rhs_columns.nonzero?
        rebuild!("numerical degradation while appending row")
      end

      @update_count += 1
      periodic_orthogonality_check!
      self
    end

    def remove_row!(index)
      normalized_index = Matrix::CoercionHelper.check_int(index, @rows.size, :index)
      snapshot = state_snapshot
      removed_row = @rows[normalized_index]

      begin
        @rows.delete_at(normalized_index)
        @rhs.delete_at(normalized_index) if @rhs_columns.nonzero?
        @frobenius_squared -= row_norm(removed_row) ** 2
        incremental_delete!(normalized_index)
      rescue NumericalRebuild
        restore_state(snapshot)
        @rows.delete_at(normalized_index)
        @rhs.delete_at(normalized_index) if @rhs_columns.nonzero?
        @frobenius_squared -= row_norm(removed_row) ** 2
        rebuild!("numerical degradation while deleting row #{normalized_index}")
      end

      @update_count += 1
      periodic_orthogonality_check!
      self
    end

    def state_snapshot
      {
        rows: deep_copy(@rows), rhs: deep_copy(@rhs), q: deep_copy(@q),
        r: deep_copy(@r), pivots: @pivots.dup, rank: @rank,
        frobenius_squared: @frobenius_squared, rebuild_count: @rebuild_count,
        last_rebuild_reason: @last_rebuild_reason,
        rhs_columns: @rhs_columns, rhs_vector: @rhs_vector
      }
    end

    def restore_state(snapshot)
      @rows = snapshot[:rows]
      @rhs = snapshot[:rhs]
      @q = snapshot[:q]
      @r = snapshot[:r]
      @pivots = snapshot[:pivots]
      @rank = snapshot[:rank]
      @frobenius_squared = snapshot[:frobenius_squared]
      @rebuild_count = snapshot[:rebuild_count]
      @last_rebuild_reason = snapshot[:last_rebuild_reason]
      @rhs_columns = snapshot[:rhs_columns]
      @rhs_vector = snapshot[:rhs_vector]
    end

    def move_last_row_to!(index)
      @rows.insert(index, @rows.pop)
      @q.insert(index, @q.pop)
      @rhs.insert(index, @rhs.pop) if @rhs_columns.nonzero?
    end

    def incremental_append!(new_row)
      @frobenius_squared += row_norm(new_row) ** 2
      row = Array.new(@column_count, 0.0)
      raise NumericalRebuild if @pivots.any?(&:nil?)
      @pivots.each_with_index { |column, i| row[i] = new_row[column] }

      @q.each { |factor_row| factor_row << 0.0 }
      @q << Array.new(@rank + 1, 0.0)
      @q.last[@rank] = 1.0
      @r << row

      @rank.times do |i|
        rotate_appended_rows!(i, @rank)
      end

      if @rank < @column_count
        candidate = (@rank...@column_count).max_by { |j| @r.last[j].abs }
        magnitude = @r.last[candidate].abs
        if magnitude > rank_tolerance &&
           magnitude > EPSILON * [matrix_scale, row_norm(new_row), 1.0].max
          unless candidate == @rank
            exchange_r_columns(candidate, @rank)
          end
          if @r.last[@rank] < 0
            flip_r_row!(@rank)
            flip_q_column!(@rank)
          end
          @rank += 1
          return
        end
      end

      @r.pop
      @q.each(&:pop)
    end

    def rotate_appended_rows!(top_index, bottom_index)
      top = @r[top_index]
      bottom = @r[bottom_index]
      a = top[top_index]
      b = bottom[top_index]
      hypotenuse = Math.hypot(a, b)
      raise NumericalRebuild if hypotenuse.zero? || !hypotenuse.finite?
      c = a / hypotenuse
      s = b / hypotenuse

      (top_index...@column_count).each do |j|
        x = top[j]
        y = bottom[j]
        top[j] = c * x + s * y
        bottom[j] = -s * x + c * y
      end

      @q.each do |factor_row|
        x = factor_row[top_index]
        y = factor_row[bottom_index]
        factor_row[top_index] = c * x + s * y
        factor_row[bottom_index] = -s * x + c * y
      end
    end

    def exchange_r_columns(first, second)
      @pivots[first], @pivots[second] = @pivots[second], @pivots[first]
      @r.each do |row|
        row[first], row[second] = row[second], row[first]
      end
    end

    def flip_r_row!(index)
      row = @r[index]
      (index...@column_count).each { |j| row[j] = -row[j] }
    end

    def flip_q_column!(index)
      @q.each { |row| row[index] = -row[index] }
    end

    def incremental_delete!(index)
      if @rows.empty?
        @q = []
        @r = []
        @rank = 0
        @pivots = (0...@column_count).to_a
        @frobenius_squared = 0.0
        return
      end

      return if @rank.zero?

      removed = @q.delete_at(index)
      (1...@rank).each do |j|
        a = removed[0]
        b = removed[j]
        hypotenuse = Math.hypot(a, b)
        next if hypotenuse.zero?
        c = a / hypotenuse
        s = b / hypotenuse
        removed[0] = c * a + s * b
        removed[j] = -s * a + c * b

        @q.each do |row|
          x = row[0]
          y = row[j]
          row[0] = c * x + s * y
          row[j] = -s * x + c * y
        end
        rotate_r_rows!(0, j, c, s)
      end

      raise NumericalRebuild unless @q.size == @rows.size
      if removed[0] > 1.0 + @orthogonality_limit
        raise NumericalRebuild
      end

      gamma = Math.sqrt(@q.sum(0.0) { |row| row[0].abs2 })
      structural_rank_loss = @rows.size < @rank
      raise NumericalRebuild if structural_rank_loss

      if gamma <= @rank_loss_limit
        delete_with_rank_loss!
      else
        if removed[0] < 0
          flip_q_column!(0)
          flip_r_row!(0)
        end
        delete_with_rank_preserved!(gamma)
      end
    end

    def rotate_r_rows!(first, second, c, s)
      first_row = @r[first]
      second_row = @r[second]
      (0...@column_count).each do |j|
        x = first_row[j]
        y = second_row[j]
        first_row[j] = c * x + s * y
        second_row[j] = -s * x + c * y
      end
    end

    def delete_with_rank_loss!
      expected_rank = @rank - 1
      if expected_rank.zero?
        @q = Array.new(@rows.size) { [] }
        @r = []
        @rank = 0
        @pivots = (0...@column_count).to_a
        return
      end

      @q = @q.map { |row| row[1...@rank] }
      @r = @r.last(expected_rank)
      @rank = expected_rank
    end

    def delete_with_rank_preserved!(gamma)
      @q.each { |row| row[0] /= gamma }
      @r[0] = @r[0].map { |value| value * gamma }
      # The transformed R is no longer upper triangular, but it is only
      # rank-by-column_count, so re-reveal its rank with a small factorization.
      transformed_r = deep_copy(@r)
      small_q, small_r, small_pivots, small_rank, =
        factorize_rows(transformed_r, rank_tolerance, @column_count)
      if small_rank != @rank
        raise NumericalRebuild
      end

      @q = matrix_multiply(@q, small_q)
      @r = small_r
      @pivots = small_pivots.map { |position| @pivots[position] }
    end

    def factorize_rows(source, tolerance, columns = @column_count)
      a = deep_copy(source)
      a.each { |row| row.concat(Array.new([columns - row.size, 0].max, 0.0)) }
      row_count = a.size
      pivots = (0...columns).to_a
      rank = 0
      frobenius_squared = source.sum(0.0) { |row| row.sum(0.0) { |value| value.abs2 } }
      q = Array.new(row_count) { [] }
      factor_r = Array.new([row_count, columns].min) { Array.new(columns, 0.0) }
      residuals = a.transpose.map(&:dup)

      while rank < [row_count, columns].min
        candidate = (rank...columns).max_by do |column|
          Math.sqrt(residuals[column].sum(0.0) { |value| value.abs2 })
        end
        candidate_vector = residuals[candidate]
        column_norm = Math.sqrt(candidate_vector.sum(0.0) { |value| value.abs2 })
        break if column_norm <= tolerance || !column_norm.finite?

        unless candidate == rank
          pivots[rank], pivots[candidate] = pivots[candidate], pivots[rank]
          a.each { |row| row[rank], row[candidate] = row[candidate], row[rank] }
          residuals[rank], residuals[candidate] = residuals[candidate], residuals[rank]
          factor_r.each { |row| row[rank], row[candidate] = row[candidate], row[rank] }
        end

        normalized = candidate_vector.map { |value| value / column_norm }
        (0...row_count).each { |i| q[i] << normalized[i] }
        rank.times { |k| factor_r[rank][k] = 0.0 }
        (rank...columns).each do |j|
          factor_r[rank][j] = normalized.each_with_index.sum(0.0) do |value, row_index|
            value * a[row_index][j]
          end
        end
        rank += 1
        residuals.each_with_index do |vector, column|
          next if column < rank
          coefficient = normalized.each_with_index.sum(0.0) do |value, row_index|
            value * vector[row_index]
          end
          normalized.each_with_index { |value, row_index| vector[row_index] -= value * coefficient }
        end
        residuals.each_with_index do |vector, column|
          next if column < rank
          vector.each_with_index { |value, row_index| a[row_index][column] = value }
        end
      end

      compact_r = rank.times.map do |i|
        Array.new(columns) do |j|
          factor_r[i][j]
        end
      end
      compact_q = q
      selected = pivots.first(rank)
      remaining = pivots[rank...columns] || []
      [compact_q, compact_r, selected + remaining, rank, frobenius_squared]
    end

    def configured_rank_tolerance(source)
      return @rank_tolerance_option if @rank_tolerance_option
      scale = Math.sqrt(source.sum(0.0) do |row|
        row.sum(0.0) { |value| value.abs2 }
      end)
      return 0.0 if scale.zero?
      EPSILON * [source.size, source.first&.size || 0, 1].max * scale
    end

    def periodic_orthogonality_check!
      return unless @orthogonality_period && @update_count.nonzero? &&
        (@update_count % @orthogonality_period).zero?
      error = factorization_error
      if !error.finite? || error > @orthogonality_limit
        rebuild!("periodic orthogonality check failed: #{error}")
      end
    end

    def factorization_error
      return 0.0 if @rows.empty? && @rank.zero?
      scaled = [1.0, matrix_scale].max
      factor_rows = rows_in_factor_order
      reconstructed = matrix_multiply(@q, @r)
      factor_error = array_error(factor_rows, reconstructed) / scaled

      return factor_error if @rank.zero?
      qtq = matrix_multiply(transpose(@q), @q)
      identity = identity_matrix(@rank)
      [factor_error, array_error(qtq, identity)].max
    end

    def rows_in_factor_order
      @rows.map { |row| @pivots.map { |column| row[column] } }
    end

    def rows_in_original_order
      deep_copy(@rows)
    end

    def minimum_norm_solution(projected)
      right_columns = projected.first&.size || @rhs_columns
      return Array.new(@column_count) { Array.new(right_columns, 0.0) } if @rank.zero?

      begin
        coefficients = if @rank == @column_count
          solve_triangular(projected)
        else
          gram = Array.new(@rank) do |i|
            Array.new(@rank) do |j|
              (0...@column_count).sum(0.0) { |column| @r[i][column] * @r[j][column] }
            end
          end
          lup = Matrix.send(:new, gram, @rank).lup
          raise NumericalRebuild if lup.singular?
          lup.solve(Matrix.send(:new, projected, right_columns)).to_a
        end
      rescue ExceptionForMatrix::ErrNotRegular
        raise NumericalRebuild
      end

      factor_order_solution = if @rank == @column_count
        coefficients
      else
        matrix_multiply(transpose(@r), coefficients)
      end
      factor_order_solution.concat(Array.new(@column_count - @rank) { Array.new(right_columns, 0.0) })
      solution = Array.new(@column_count)
      @pivots.each_with_index { |original_column, factor_column|
        solution[original_column] = factor_order_solution[factor_column]
      }
      solution
    end

    def solve_triangular(projected)
      right_columns = projected.first.size
      coefficients = Array.new(@rank) { Array.new(right_columns, 0.0) }
      (@rank - 1).downto(0) do |i|
        pivot = @r[i][i]
        raise NumericalRebuild if pivot.abs <= rank_tolerance
        right_columns.times do |j|
          value = projected[i][j] - ((i + 1)...@rank).sum(0.0) do |k|
            @r[i][k] * coefficients[k][j]
          end
          coefficients[i][j] = value / pivot
        end
      end
      coefficients
    end

    def transpose_multiply(left, right)
      inner = left.first&.size || 0
      columns = right.first&.size || 0
      Array.new(inner) do |i|
        Array.new(columns) do |j|
          left.each_with_index.sum(0.0) { |row, k| row[i] * right[k][j] }
        end
      end
    end

    def matrix_multiply(left, right)
      rows = left.size
      inner = right.size
      columns = right.first&.size || 0
      Array.new(rows) do |i|
        Array.new(columns) do |j|
          (0...inner).sum(0.0) { |k| left[i][k] * right[k][j] }
        end
      end
    end

    def matrix_subtract(left, right)
      left.each_with_index.map do |row, i|
        row.each_with_index.map { |value, j| value - right[i][j] }
      end
    end

    def transpose(matrix)
      columns = matrix.first&.size || 0
      Array.new(columns) do |column|
        matrix.map { |row| row[column] }
      end
    end

    def array_error(left, right)
      squared = 0.0
      left.each_with_index do |row, i|
        row.each_with_index do |value, j|
          squared += (value - right[i][j]).abs2
        end
      end
      Math.sqrt(squared)
    end

    def column_norms(matrix)
      columns = matrix.first&.size || 0
      Array.new(columns) do |column|
        Math.sqrt(matrix.sum(0.0) { |row| row[column].abs2 })
      end
    end

    def frobenius_norm(matrix)
      Math.sqrt(matrix.sum(0.0) do |row|
        row.sum(0.0) { |value| value.abs2 }
      end)
    end

    def matrix_scale
      Math.sqrt(@frobenius_squared)
    end

    def row_norm(row)
      Math.sqrt(row.sum(0.0) { |value| value.abs2 })
    end

    def identity_matrix(size)
      Array.new(size) do |i|
        Array.new(size) { |j| i == j ? 1.0 : 0.0 }
      end
    end

    def deep_copy(matrix)
      matrix.map(&:dup)
    end

    def column_to_vector(matrix, column)
      Vector.elements(matrix.map { |row| row[column] }, false)
    end
  end
end
