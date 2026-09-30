# frozen_string_literal: false
class Matrix
  # Real Schur decomposition +A = Q T Q^T+ with optional spectral reordering.
  # Q is orthogonal and T is real quasi-upper-triangular.
  class RealSchurDecomposition
    DimensionMismatch = ExceptionForMatrix::ErrDimensionMismatch
    class ReorderError < StandardError
      attr_reader :reason

      def initialize(reason, message)
        @reason = reason
        super(message)
      end
    end

    attr_reader :q, :t, :selected_dimension, :block_sizes, :eigenvalue_blocks

    alias_method :orthogonal_matrix, :q
    alias_method :schur_matrix, :t
    alias_method :m, :selected_dimension
    alias_method :selected_subspace_dimension, :selected_dimension

    def initialize(matrix, selector = nil, by: nil, real_part: nil, magnitude: nil, select: nil, &predicate)
      unless matrix.is_a?(Matrix)
        raise TypeError, "Expected Matrix but got #{matrix.class}"
      end
      raise DimensionMismatch unless matrix.square?
      unless matrix.all? { |entry| entry.is_a?(Numeric) && !entry.is_a?(Complex) && entry.finite? }
        raise ArgumentError, "Real Schur decomposition requires a finite real square matrix"
      end

      size = matrix.row_count
      @source = Matrix.send(:new, matrix.to_a.map { |row| row.map(&:to_f) }, size)
      if size.zero?
        @q = Matrix.send(:new, [], 0)
        @t = @q.dup
        @block_sizes = []
        @eigenvalue_blocks = []
        @selected_dimension = 0
        return
      end
      eigensystem = EigenvalueDecomposition.new(matrix, false)
      q_rows = eigensystem.instance_variable_get(:@v).map { |row| row.map(&:to_f) }
      @q = Matrix.send(:new, q_rows)
      schur_rows = eigensystem.instance_variable_get(:@h)
      if schur_rows
        @t = Matrix.send(:new, schur_rows.map { |row| row.map(&:to_f) })
      else
        input = Matrix.send(:new, matrix.to_a.map { |row| row.map(&:to_f) })
        @t = @q.transpose * input * @q
      end

      @block_sizes, @eigenvalue_blocks = block_information(@t)
      @selected_dimension = 0
      return if size.zero?

      criterion, target = resolve_selector(selector, by, real_part, magnitude, select, predicate)
      selected = @eigenvalue_blocks.map { |block| block_selected?(block, criterion, target) }
      original_block_sizes = @block_sizes.dup
      reorder(selected)
      @block_sizes, @eigenvalue_blocks = block_information(@t)
      @selected_dimension = selected.each_with_index.sum { |chosen, index| chosen ? original_block_sizes[index] : 0 }
    end

    def eigenvalues
      @eigenvalue_blocks.flatten
    end

    def to_ary
      [@q, @t]
    end
    alias_method :to_a, :to_ary

    private

    def resolve_selector(selector, by, real_part, magnitude, select, predicate)
      if selector.is_a?(Hash)
        by ||= selector.fetch(:by, selector.fetch(:criterion, selector.fetch(:measure, nil)))
        selector = selector.fetch(:select, selector.fetch(:value, selector.fetch(:magnitude, selector.fetch(:real_part, nil))))
      end

      target = select || selector
      criterion = by
      if real_part
        criterion = :real_part
        target = real_part
      elsif magnitude
        criterion = :magnitude
        target = magnitude
      end

      return [:predicate, predicate] if predicate
      return [:none, nil] if target.nil?
      return [:predicate, target] if target.respond_to?(:call) && criterion.nil? && !selector.is_a?(Range)

      criterion = :real_part if criterion.nil? && selector.is_a?(Range)
      [check_criterion(criterion), target]
    end

    def check_criterion(criterion)
      case criterion
      when :real_part, :real, :re
        :real_part
      when :magnitude, :modulus, :abs, :absolute
        :magnitude
      else
        raise ArgumentError, "unsupported Schur selection criterion #{criterion.inspect}"
      end
    end

    def block_selected?(block, criterion, target)
      case criterion
      when :none
        false
      when :predicate
        result = target.arity == 1 ? target.call(block.first) : target.call(*block)
        !!result
      when :real_part
        target === block.first.real
      else
        target === block.first.abs
      end
    end

    def reorder(selected)
      order = selected.each_index.to_a
      order.each_with_index do |block_index, position|
        next unless selected[block_index]

        current = position
        while current > 0 && !selected[order[current - 1]]
          left_index = order[current - 1]
          right_index = order[current]
          start = order[0...current - 1].sum { |index| @block_sizes[index] }
          upper = start + @block_sizes[left_index] + @block_sizes[right_index]
          swap_adjacent_blocks(@block_sizes[left_index], @block_sizes[right_index], start, upper)
          order[current - 1], order[current] = order[current], order[current - 1]
          current -= 1
        end
      end
    end

    def swap_adjacent_blocks(left_size, right_size, lower, upper)
      local_size = left_size + right_size
      local = Array.new(local_size) { |i| Array.new(local_size) { |j| @t[lower + i, lower + j] } }
      solution = nil
      orthogonal = swap_schur_rotation(local, nil, left_size, right_size) if left_size == 1 && right_size == 1
      unless orthogonal
        rhs = Array.new(left_size) { |i| Array.new(right_size) { |j| -local[i][left_size + j] } }
        solution = solve_sylvester(local, rhs, left_size, right_size)
        orthogonal = swap_schur_rotation(local, solution, left_size, right_size)
      end
      transformed = multiply_orthogonal(local, orthogonal)
      replace_local_schur(transformed, orthogonal, lower, upper)
    end

    def swap_schur_rotation(local, solution, left_size, right_size)
      return householder_qr(build_swap_basis(solution, left_size, right_size)) unless left_size == 1 && right_size == 1

      size = left_size + right_size
      rotation = Array.new(size) { |i| Array.new(size) { |j| i == j ? 1.0 : 0.0 } }
      gap = local[1][1] - local[0][0]
      tolerance = Math.sqrt(Float::EPSILON) * [1.0, local.flatten.map(&:abs).max].max
      coupling = local[0][1].abs >= local[1][0].abs ? local[0][1] : -local[1][0]
      if gap.abs <= tolerance && coupling.abs > tolerance
        raise ReorderError.new(:indistinguishable_eigenvalues,
          "Schur blocks cannot be reliably separated near repeated eigenvalues")
      end
      x = gap.zero? ? 0.0 : coupling / gap
      if x.abs > 1.0 / Math.sqrt(Float::EPSILON)
        raise ReorderError.new(:indistinguishable_eigenvalues,
          "Schur blocks cannot be reliably separated near repeated eigenvalues")
      end
      radius = Math.hypot(x, 1.0)
      rotation[0][0] = x / radius
      rotation[0][1] = 1.0 / radius
      rotation[1][0] = 1.0 / radius
      rotation[1][1] = -x / radius
      rotation
    end

    def solve_sylvester(local, rhs, left_size, right_size)
      unknown_count = left_size * right_size
      vector = Array.new(unknown_count)
      matrix = Array.new(unknown_count) { Array.new(unknown_count, 0.0) }

      right_size.times do |right_column|
        left_size.times do |right_row|
          row = right_column * left_size + right_row
          vector[row] = rhs[right_row][right_column]
          left_size.times do |left_column|
            matrix[row][right_column * left_size + left_column] += local[right_row][left_column]
          end
          right_size.times do |left_column|
            matrix[row][left_column * left_size + right_row] -= local[left_size + right_column][left_size + left_column]
          end
        end
      end

      return Array.new(left_size) { Array.new(right_size, 0.0) } if vector.all?(&:zero?)

      pivot_scale = [1.0, local.flatten.map(&:abs).max].max
      singular_limit = Math.sqrt(Float::EPSILON) * pivot_scale
      unknown_count.times do |column|
        pivot = (column...unknown_count).max_by { |row| matrix[row][column].abs }
        if matrix[pivot][column].abs <= singular_limit
          raise ReorderError.new(:indistinguishable_eigenvalues,
            "Schur blocks cannot be reliably separated near repeated eigenvalues")
        end
        matrix[pivot], matrix[column] = matrix[column], matrix[pivot]
        vector[pivot], vector[column] = vector[column], vector[pivot]

        (column + 1...unknown_count).each do |row|
          factor = matrix[row][column] / matrix[column][column]
          (column + 1...unknown_count).each { |j| matrix[row][j] -= factor * matrix[column][j] }
          vector[row] -= factor * vector[column]
        end
      end

      solution = Array.new(unknown_count, 0.0)
      (unknown_count - 1).downto(0) do |row|
        solution[row] = vector[row]
        (row + 1...unknown_count).each { |column| solution[row] -= matrix[row][column] * solution[column] }
        solution[row] /= matrix[row][row]
      end

      result = Array.new(left_size) { |i| Array.new(right_size) { |j| solution[j * left_size + i] } }
      if result.flatten.map(&:abs).max > 1.0 / (Math.sqrt(Float::EPSILON) * pivot_scale)
        raise ReorderError.new(:indistinguishable_eigenvalues,
          "Schur blocks cannot be reliably separated near repeated eigenvalues")
      end
      result
    end

    def build_swap_basis(solution, left_size, right_size)
      size = left_size + right_size
      basis = Array.new(size) do |i|
        i < left_size ? solution[i].dup : Array.new(right_size, 0.0)
      end
      right_size.times { |i| basis[left_size + i][i] = 1.0 }
      basis
    end

    def householder_qr(matrix)
      row_count = matrix.size
      column_count = matrix.first.size
      working = matrix.map(&:dup)
      orthogonal = Array.new(row_count) { |i| Array.new(row_count) { |j| i == j ? 1.0 : 0.0 } }
      transformations = []

      column_count.times do |column|
        vector = Array.new(row_count, 0.0)
        (column...row_count).each { |i| vector[i] = working[i][column] }
        norm = Math.sqrt(vector.sum { |value| value**2 })
        next if norm.zero?

        vector[column] += working[column][column] >= 0 ? -norm : norm
        vector_norm2 = vector.sum { |value| value**2 }
        column_dots = Array.new(column_count) { |j| (column...row_count).sum { |i| vector[i] * (j >= column ? working[i][j] : 0.0) } }
        column_count.times do |j|
          next if j < column
          (column...row_count).each do |row|
            working[row][j] -= 2.0 * column_dots[j] / vector_norm2 * vector[row]
          end
        end
        transformations << [vector, vector_norm2, column]
      end

      transformations.reverse_each do |vector, vector_norm2, column|
        orthogonal = orthogonal.map(&:dup)
        q_dots = Array.new(row_count) { |j| (column...row_count).sum { |i| vector[i] * orthogonal[i][j] } }
        (column...row_count).each do |row|
          next if vector[row].zero?
          row_count.times { |j| orthogonal[row][j] -= 2.0 * q_dots[j] / vector_norm2 * vector[row] }
        end
      end
      orthogonal
    end

    def multiply_orthogonal(matrix, orthogonal)
      size = matrix.size
      temporary = Array.new(size) do |i|
        Array.new(size) { |j| (0...size).sum { |k| orthogonal[k][i] * matrix[k][j] } }
      end
      Array.new(size) do |i|
        Array.new(size) { |j| (0...size).sum { |k| temporary[i][k] * orthogonal[k][j] } }
      end
    end

    def replace_local_schur(local, orthogonal, lower, upper)
      t_rows = @t.to_a
      q_rows = @q.to_a
      local_size = upper - lower

      local_size.times do |i|
        local_size.times { |j| t_rows[lower + i][lower + j] = local[i][j] }
      end

      new_q = q_rows.map.with_index do |row, row_index|
        updated = row.dup
        if local_size == 2
          gap = local[1][1] - local[0][0]
          coupling = local[0][1].abs >= local[1][0].abs ? local[0][1] : -local[1][0]
          x = gap.zero? ? 0.0 : coupling / gap
          radius = Math.hypot(x, 1.0)
          first = x * row[lower] + row[lower + 1]
          second = row[lower] - x * row[lower + 1]
          updated[lower] = first / radius
          updated[lower + 1] = second / radius
        else
        local_size.times do |local_column|
          updated[lower + local_column] = (lower...upper).sum do |column|
            q_rows[row_index][column] * orthogonal[column - lower][local_column]
          end
        end
        end
        updated
      end

      @q = Matrix.send(:new, new_q)
      @t = @q.transpose * @source * @q
    end

    def block_information(matrix)
      size = matrix.row_count
      return [[], []] if size.zero?

      tolerance = 1000.0 * Float::EPSILON * [1.0, matrix.map(&:abs).max].max
      sizes = []
      values = []
      i = 0
      while i < size
        if i + 1 < size && matrix[i + 1, i].abs > tolerance
          a = matrix[i, i]
          b = matrix[i, i + 1]
          c = matrix[i + 1, i]
          d = matrix[i + 1, i + 1]
          real = (a + d) / 2.0
          discriminant = (a - d)**2 + 4.0 * b * c
          imaginary = Math.sqrt([-discriminant, 0.0].max) / 2.0
          sizes << 2
          values << [Complex(real, imaginary), Complex(real, -imaginary)]
          i += 2
        else
          sizes << 1
          values << [matrix[i, i].to_f]
          i += 1
        end
      end
      [sizes, values]
    end

  end
end
