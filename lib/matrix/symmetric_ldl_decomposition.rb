# frozen_string_literal: false
class Matrix
  #
  # For a symmetric matrix +A+ with exact +Integer+ or +Rational+ entries,
  # SymmetricLDLDecomposition computes a pivot-permuted block LDL decomposition
  #
  #   P * A * P.t = L * D * L.t
  #
  # where +P+ is a permutation matrix and +L+ is unit lower triangular.
  # Equivalently, using the inverse permutation matrix,
  #
  #   A = P.t * L * D * L.t * P.
  #
  # +D+ is symmetric and block diagonal. Each pivot block is either one by one
  # or has the form [[0, b], [b, 0]] with nonzero +b+. Trailing zero rows and
  # columns are represented as one-by-one zero blocks in +D+, so the
  # decomposition remains valid when +A+ is rank deficient.
  #
  class SymmetricLDLDecomposition
    # Returns the permutation matrix +P+
    attr_reader :p

    # Returns the unit lower triangular factor +L+
    attr_reader :l

    # Returns the block diagonal factor +D+
    attr_reader :d

    # Returns the pivot indices such that +p[i][pivots[i]] == 1+
    attr_reader :pivots

    # Returns the ordered pivot block sizes (+1+ or +2+)
    attr_reader :block_sizes

    # Returns the number of positive eigenvalues of +A+
    attr_reader :positive_count

    # Returns the number of negative eigenvalues of +A+
    attr_reader :negative_count

    # Returns the number of zero eigenvalues of +A+
    attr_reader :zero_count

    # Returns +P+, +L+, and +D+ in an array
    def to_ary
      [p, l, d]
    end
    alias_method :to_a, :to_ary

    # Returns the inertia as +[positive_count, negative_count, zero_count]+
    def inertia
      [positive_count, negative_count, zero_count]
    end

    # Returns the rank of +A+
    attr_reader :rank

    # Returns +true+ if +D+, and hence +A+, is singular
    def singular?
      zero_count != 0
    end

    # Returns an exact basis for the null space of +A+ as columns of a matrix
    def nullspace
      columns = Array.new(@size) { Array.new(@nullspace_vectors.size, 0) }
      @nullspace_vectors.each_with_index do |vector, column|
        vector.each_with_index do |value, row|
          columns[@pivots[row]][column] = value
        end
      end
      Matrix.send :new, columns, @nullspace_vectors.size
    end

    def initialize(a)
      raise TypeError, "Expected Matrix but got #{a.class}" unless a.is_a?(Matrix)
      raise Matrix::ErrDimensionMismatch unless a.square?
      raise ArgumentError, "SymmetricLDLDecomposition requires a symmetric matrix" unless a.symmetric?
      unless a.all? {|entry| entry.is_a?(Integer) || entry.is_a?(Rational)}
        raise TypeError, "SymmetricLDLDecomposition requires Integer or Rational entries"
      end

      @size = a.row_count
      @work = a.to_a
      @pivots = (0...@size).to_a
      @l_rows = Array.new(@size) {|i| Array.new(@size) {|j| i == j ? 1 : 0} }
      @d_rows = Array.new(@size) { Array.new(@size, 0) }
      @block_starts = []
      @block_sizes = []
      @positive_count = 0
      @negative_count = 0
      @rank = 0

      k = 0
      while k < @size
        pivot = (k...@size).find {|i| @work[i][i] != 0}

        if pivot
          exchange_rows_and_columns(k, pivot) if pivot != k
          decompose_one_by_one(k)
          k += 1
        else
          pair = find_two_by_two_pivot(k)
          break unless pair

          first, second = pair
          exchange_rows_and_columns(k, first) if first != k
          if second != k + 1
            exchange_rows_and_columns(k + 1, second)
          end
          decompose_two_by_two(k)
          k += 2
        end
      end

      @zero_count = @size - k
      @l = Matrix.send :new, @l_rows, @size
      @d = Matrix.send :new, @d_rows, @size
      @p = build_permutation
      @nullspace_vectors = build_nullspace_vectors(k)
    end

    private

    def exchange_rows_and_columns(first, second)
      @work[first], @work[second] = @work[second], @work[first]
      @size.times do |i|
        @work[i][first], @work[i][second] = @work[i][second], @work[i][first]
      end

      @l_rows[first], @l_rows[second] = @l_rows[second], @l_rows[first]
      @l_rows.each do |row|
        row[first], row[second] = row[second], row[first]
      end

      first_original = @pivots[first]
      @pivots[first] = @pivots[second]
      @pivots[second] = first_original

    end

    def find_two_by_two_pivot(k)
      (k...@size).each do |i|
        (i + 1...@size).each do |j|
          return [i, j] if @work[i][j] != 0
        end
      end
      nil
    end

    def decompose_one_by_one(k)
      pivot = @work[k][k]
      @block_starts << k
      count_pivot(pivot)
      @d_rows[k][k] = pivot
      @block_sizes << 1
      @rank += 1

      (k + 1...@size).each do |i|
        multiplier = @work[i][k].quo(pivot)
        @l_rows[i][k] = multiplier
        (i...@size).each do |j|
          value = @work[i][j] - multiplier * @work[k][j]
          @work[i][j] = value
          @work[j][i] = value
        end
      end
    end

    def decompose_two_by_two(k)
      off_diagonal = @work[k][k + 1]
      @block_starts << k
      @d_rows[k][k + 1] = off_diagonal
      @d_rows[k + 1][k] = off_diagonal
      @block_sizes << 2
      @rank += 2
      @positive_count += 1
      @negative_count += 1

      (k + 2...@size).each do |i|
        first_multiplier = @work[i][k + 1].quo(off_diagonal)
        second_multiplier = @work[i][k].quo(off_diagonal)
        @l_rows[i][k] = first_multiplier
        @l_rows[i][k + 1] = second_multiplier

        (i...@size).each do |j|
          value = @work[i][j] -
                  first_multiplier * @work[k][j] -
                  second_multiplier * @work[k + 1][j]
          @work[i][j] = value
          @work[j][i] = value
        end
      end
    end

    def count_pivot(pivot)
      if pivot > 0
        @positive_count += 1
      else
        @negative_count += 1
      end
    end

    def build_permutation
      rows = Array.new(@size) { Array.new(@size, 0) }
      @pivots.each_with_index do |original_row, row|
        rows[row][original_row] = 1
      end
      Matrix.send :new, rows, @size
    end

    def build_nullspace_vectors(pivot_end)
      vectors = []
      (pivot_end...@size).each do |zero_index|
        vector = Array.new(@size, 0)
        vector[zero_index] = 1

        @block_starts.reverse.each_with_index do |block_start, reverse_index|
          block_size = @block_sizes[@block_sizes.size - reverse_index - 1]
          block_end = block_start + block_size

          if block_size == 1
            sum = (block_end...@size).sum {|j| @l_rows[j][block_start] * vector[j]}
            vector[block_start] = -sum
          else
            last = block_end - 1
            end_sum = (block_end...@size).sum {|j| @l_rows[j][last] * vector[j]}
            start_sum = (block_end...@size).sum {|j| @l_rows[j][block_start] * vector[j]}
            vector[last] = -end_sum
            vector[block_start] = -start_sum
          end
        end

        vectors << vector
      end
      vectors
    end
  end
end
