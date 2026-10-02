# frozen_string_literal: false

class Matrix
  # Exact symmetric LDL^T decomposition with Bunch-Parlett style pivoting.
  #
  # Given a symmetric matrix +A+ with Integer or Rational entries, this computes
  # a permutation matrix +P+, a unit lower triangular matrix +L+, and a block
  # diagonal symmetric matrix +D+ such that:
  #
  #   P * A * P.transpose = L * D * L.transpose
  #
  # Nonzero blocks of +D+ are 1x1 pivots or 2x2 pivots of the form
  # +[[0, b], [b, 0]]+. A zero diagonal entry therefore does not imply that the
  # whole matrix is singular. Any remaining zero directions are retained as a
  # zero trailing block, together with an exact nullspace basis.
  #
  # Inertia is invariant under nonsingular congruence: each 1x1 pivot
  # contributes one positive or negative direction, and each nonzero 2x2 pivot
  # contributes one positive and one negative direction.
  class SymmetricLDLDecomposition
    # Array of pivot indices: +pivots[i]+ is the original matrix row/column
    # moved to position +i+. +pivot_block_sizes+ lists the 1x1 and 2x2 blocks
    # along the diagonal of +d+ (with trailing 1x1 zero blocks).
    attr_reader :pivots, :pivot_block_sizes, :rank, :inertia

    # Returns the unit lower triangular factor +L+.
    def l
      Matrix.send(:new, @l_rows.map(&:dup), @size)
    end

    # Returns the block diagonal symmetric factor +D+.
    def d
      Matrix.send(:new, @d_rows.map(&:dup), @size)
    end

    # Returns the permutation matrix +P+; +pivots+ gives the same permutation
    # as row/column indices.
    def p
      rows = Array.new(@size) { Array.new(@size, 0) }
      @pivots.each_with_index { |pivot, row| rows[row][pivot] = 1 }
      Matrix.send(:new, rows, @size)
    end

    def to_ary
      [p, l, d]
    end
    alias_method :to_a, :to_ary

    # Number of positive directions (positive eigenvalues by Sylvester's law).
    def positive_count
      @inertia.fetch(:positive)
    end

    # Number of negative directions (negative eigenvalues by Sylvester's law).
    def negative_count
      @inertia.fetch(:negative)
    end

    # Number of zero directions (nullity of the matrix).
    def zero_count
      @inertia.fetch(:zero)
    end
    alias_method :nullity, :zero_count

    # Returns an exact matrix whose columns form a basis of the nullspace.
    def nullspace
      return Matrix.empty(@size, 0) if @rank == @size

      columns = Array.new(@size - @rank) do |column|
        vector = Array.new(@size, 0)
        zero_index = @rank + column
        vector[zero_index] = 1

        (@rank - 1).downto(0) do |row|
          vector[row] = -(row + 1 ... @size).sum { |column_index|
            @l_rows[column_index][row] * vector[column_index]
          }
        end

        original = Array.new(@size, 0)
        @pivots.each_with_index { |pivot, index| original[pivot] = vector[index] }
        original
      end

      Matrix.columns(columns)
    end

    def initialize(matrix)
      unless matrix.is_a?(Matrix)
        raise TypeError, "Expected Matrix but got #{matrix.class}"
      end
      raise Matrix::ErrDimensionMismatch unless matrix.square?
      raise Matrix::ErrNotSymmetric unless matrix.symmetric?

      matrix.each do |entry|
        unless entry.is_a?(Integer) || entry.is_a?(Rational)
          raise TypeError, "symmetric LDL decomposition requires Integer or Rational entries"
        end
      end

      @size = matrix.row_count
      @pivots = Array.new(@size, &:itself)
      @pivot_block_sizes = []
      @work = matrix.to_a.map { |row| row.map { |entry| Rational(entry) } }
      @d_rows = Array.new(@size) { Array.new(@size, 0) }
      @l_rows = Array.new(@size) do |row|
        Array.new(@size) { |column| row == column ? 1 : 0 }
      end
      @rank = 0
      @positive = 0
      @negative = 0

      factor

      @zero_count = @size - @rank
      @inertia = {
        positive: @positive,
        negative: @negative,
        zero: @zero_count
      }.freeze
      @pivot_block_sizes.concat(Array.new(@zero_count, 1))
      @pivots.freeze
      @pivot_block_sizes.freeze
    end

    private

    def factor
      index = 0

      while index < @size
        one_by_one_pivot = (index ... @size).find do |candidate|
          @work[candidate][candidate] != 0
        end

        if one_by_one_pivot
          swap_active(index, one_by_one_pivot) unless one_by_one_pivot == index
          factor_one_by_one(index)
          @pivot_block_sizes << 1
          @rank += 1
          if @work[index][index] > 0
            @positive += 1
          else
            @negative += 1
          end
          index += 1
          next
        end

        pair = find_two_by_two_pivot(index)
        break unless pair

        first, second = pair
        swap_active(index, first) unless first == index
        swap_active(index + 1, second) unless second == index + 1
        factor_two_by_two(index)
        @pivot_block_sizes << 2
        @rank += 2
        @positive += 1
        @negative += 1
        index += 2
      end
    end

    def find_two_by_two_pivot(index)
      (index ... @size).each do |row|
        (row + 1 ... @size).each do |column|
          return [row, column] if @work[row][column] != 0
        end
      end
      nil
    end

    def swap_active(first, second)
      return if first == second

      @work[first], @work[second] = @work[second], @work[first]
      @work.each do |row|
        row[first], row[second] = row[second], row[first]
      end
      @l_rows[first], @l_rows[second] = @l_rows[second], @l_rows[first]
      @l_rows.each do |row|
        row[first], row[second] = row[second], row[first]
      end
      @pivots[first], @pivots[second] = @pivots[second], @pivots[first]
    end

    def factor_one_by_one(index)
      pivot = @work[index][index]

      multipliers = []
      (index + 1 ... @size).each do |row|
        multiplier = @work[row][index].quo(pivot)
        @l_rows[row][index] = multiplier
        multipliers << multiplier
      end
      @d_rows[index][index] = pivot

      (index + 1 ... @size).each do |row|
        multiplier = multipliers[row - index - 1]
        @work[row][index] = 0
        @work[index][row] = 0
        (row ... @size).each do |column|
          column_multiplier = multipliers[column - index - 1]
          updated = @work[row][column] - pivot * multiplier * column_multiplier
          @work[row][column] = updated
          @work[column][row] = updated
        end
      end
    end

    def factor_two_by_two(index)
      off_diagonal = @work[index][index + 1]
      @d_rows[index][index + 1] = off_diagonal
      @d_rows[index + 1][index] = off_diagonal

      first_multipliers = []
      second_multipliers = []
      (index + 2 ... @size).each do |row|
        first_multiplier = @work[row][index + 1].quo(off_diagonal)
        second_multiplier = @work[row][index].quo(off_diagonal)
        @l_rows[row][index] = first_multiplier
        @l_rows[row][index + 1] = second_multiplier
        first_multipliers << first_multiplier
        second_multipliers << second_multiplier
      end

      (index + 2 ... @size).each do |row|
        first_multiplier = first_multipliers[row - index - 2]
        second_multiplier = second_multipliers[row - index - 2]
        @work[row][index] = 0
        @work[index][row] = 0
        @work[row][index + 1] = 0
        @work[index + 1][row] = 0
        (row ... @size).each do |column|
          if column < index + 2 + first_multipliers.size
            column_first_multiplier = first_multipliers[column - index - 2]
            column_second_multiplier = second_multipliers[column - index - 2]
            updated =
              @work[row][column] -
              off_diagonal * first_multiplier * column_second_multiplier -
              off_diagonal * second_multiplier * column_first_multiplier
          else
            updated = @work[row][column]
          end
          @work[row][column] = updated
          @work[column][row] = updated
        end
      end
    end
  end
end
