# frozen_string_literal: false
class Matrix
  #
  # For a doubly stochastic matrix +A+ (a square matrix of non-negative
  # entries whose rows and columns all sum to 1), the Birkhoff–von Neumann
  # theorem states that +A+ is a convex combination of permutation matrices:
  #
  #   A = w1*P1 + w2*P2 + ... + wk*Pk,   w1 + w2 + ... + wk = 1,  wi > 0
  #
  # The decomposition is computed exactly: entries must be Integer or
  # Rational, and each permutation only uses positions where the
  # corresponding entry of +A+ is positive, so the weighted sum of the
  # returned permutations reproduces +A+ exactly.
  #
  # The permutations are found by repeated bipartite perfect matching on
  # the support of the remaining matrix, so at most n*n - n + 1
  # permutations are produced, without enumerating all n! permutations.
  #
  class BirkhoffDecomposition
    include Enumerable

    # Returns the permutations of the decomposition, one per term.
    # Each permutation is an Array of +column_count+ column indices,
    # one for each row.

    attr_reader :permutations

    # Returns the positive Rational weights of the decomposition,
    # in the same order as the permutations. They sum to exactly 1.

    attr_reader :weights

    # Yields each +[permutation, weight]+ pair of the decomposition.

    def each(&block)
      return to_enum(__method__) { @pairs.size } unless block_given?
      @pairs.each(&block)
      self
    end

    # Returns the decomposition as an Array of +[permutation, weight]+ pairs.

    def to_ary
      @pairs.map { |permutation, weight| [permutation.dup, weight] }
    end
    alias_method :to_a, :to_ary

    def initialize(m)
      raise TypeError, "Expected Matrix but got #{m.class}" unless m.is_a?(Matrix)
      raise Matrix::ErrDimensionMismatch unless m.square?
      n = m.row_count
      remaining = check_doubly_stochastic(m, n)
      @pairs = decompose(remaining, n)
      @permutations = @pairs.map { |permutation, _| permutation.freeze }.freeze
      @weights = @pairs.map { |_, weight| weight }.freeze
    end

    private

    # Validates that +m+ is doubly stochastic and returns its entries as
    # a mutable Array of Arrays of Rationals.
    def check_doubly_stochastic(m, n)
      entries = Array.new(n) { Array.new(n) }
      m.each_with_index do |e, i, j|
        unless e.is_a?(Integer) || e.is_a?(Rational)
          raise Matrix::ErrNotDoublyStochastic,
                "element (#{i}, #{j}) is not an Integer or Rational: #{e.inspect}"
        end
        if e < 0
          raise Matrix::ErrNotDoublyStochastic,
                "element (#{i}, #{j}) is negative: #{e.inspect}"
        end
        entries[i][j] = Rational(e)
      end
      n.times do |i|
        sum = entries[i].sum
        if sum != 1
          raise Matrix::ErrNotDoublyStochastic,
                "row #{i} sums to #{sum.inspect}, expected 1"
        end
      end
      n.times do |j|
        sum = (0...n).sum { |i| entries[i][j] }
        if sum != 1
          raise Matrix::ErrNotDoublyStochastic,
                "column #{j} sums to #{sum.inspect}, expected 1"
        end
      end
      entries
    end

    # Repeatedly extracts a permutation from the support of the remaining
    # matrix and subtracts its smallest entry, until nothing remains.
    def decompose(remaining, n)
      pairs = []
      while n > 0 && (permutation = perfect_matching(remaining, n))
        weight = (0...n).map { |i| remaining[i][permutation[i]] }.min
        n.times { |i| remaining[i][permutation[i]] -= weight }
        pairs << [permutation, weight]
      end
      pairs
    end

    # Finds a perfect matching in the bipartite graph of the positive
    # entries of +remaining+ (Kuhn's augmenting path algorithm), or +nil+
    # if none exists (i.e. the remaining matrix is zero).
    def perfect_matching(remaining, n)
      match_row = Array.new(n) # row    -> column
      match_col = Array.new(n) # column -> row
      n.times do |row|
        return nil unless augment(row, remaining, n, Array.new(n, false), match_row, match_col)
      end
      match_row
    end

    def augment(row, remaining, n, seen, match_row, match_col)
      n.times do |col|
        next if seen[col] || remaining[row][col].zero?
        seen[col] = true
        owner = match_col[col]
        if owner.nil? || augment(owner, remaining, n, seen, match_row, match_col)
          match_row[row] = col
          match_col[col] = row
          return true
        end
      end
      false
    end
  end
end
