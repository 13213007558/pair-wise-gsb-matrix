# frozen_string_literal: false
class Matrix
  # Maintains a QR factorization of a sliding row window.
  #
  # Rows are grouped into immutable blocks of at least one and at most
  # +block_size+ rows.  Each block has a QR factorization.  A balanced binary
  # tree combines block factorizations; updating one row therefore recomputes
  # one small block and O(log(m/block_size)) pair factorizations, without ever
  # rebuilding the complete m-by-n matrix.

  class UpdatableQRDecomposition
    DEFAULT_BLOCK_SIZE = 32
    DEFAULT_REBUILD_TOLERANCE = 1e-12
    MINIMUM_NORM_SOLUTION = "minimum-norm least-squares solution"
    UNIQUE_SOLUTION = "unique least-squares solution"

    include Matrix::ConversionHelper

    Node = Struct.new(:rows, :left, :right, :height,
                      :r, :pivoted_r, :pivots, :rank,
                      :pivoted_y, :full_b, :pivoted_full_b, :row_count, :leaf) # :nodoc:

    class << self
      # Creates an empty updatable QR factorization.  If +column_count+ is
      # omitted, it is inferred from the first appended row.
      def empty(column_count = nil, rhs_count = 0, **options)
        allocate.tap do |qr|
          qr.send(:initialize_empty, column_count, rhs_count, options)
        end
      end
    end

    # The number of columns in the row window.
    attr_reader :column_count

    # The number of right-hand-side columns currently maintained with A.
    attr_reader :rhs_count

    # Maximum number of original rows retained in one leaf block.
    attr_reader :block_size

    # Absolute tolerance used to decide numerical rank.  +nil+ means an
    # automatic tolerance is chosen for every QR factorization.
    attr_reader :rank_tolerance

    # Relative error allowed before accumulated factors are discarded.
    attr_reader :rebuild_tolerance

    # Number of times a complete factor rebuild was triggered by checks.
    attr_reader :rebuild_count

    # Reason for the most recent complete rebuild.
    attr_reader :last_rebuild_reason

    # Creates an updatable QR factorization for +matrix+.  +rhs+ may be a Vector,
    # an Array, or a Matrix with one right-hand side per column.
    def initialize(matrix = nil, rhs = nil, block_size: DEFAULT_BLOCK_SIZE,
                   rank_tolerance: nil, rebuild_tolerance: DEFAULT_REBUILD_TOLERANCE)
      if matrix.nil?
        initialize_empty(nil, 0, block_size: block_size,
                         rank_tolerance: rank_tolerance,
                         rebuild_tolerance: rebuild_tolerance)
        return
      end

      matrix = CoercionHelper.coerce_to_matrix(matrix)
      rows = matrix.to_a
      initialize_state(matrix.column_count, block_size, rank_tolerance,
                        rebuild_tolerance)
      values = initialize_rhs(rhs, matrix.row_count)
      rows.each_with_index do |row, index|
        append_checked_row(validated_row(row), values ? values[index] : [])
      end
    end

    # Number of rows currently represented by A.
    def row_count
      @row_count
    end
    alias_method :row_size, :row_count

    # Number of right-hand-side columns; 1 when a vector is being tracked.
    alias_method :rhs_column_count, :rhs_count

    # Appends +row+ with the corresponding right-hand-side value(s).
    def append_row(row, rhs)
      row = validated_row(row)
      rhs_values = validated_rhs_value(rhs)
      append_checked_row(row, rhs_values)
      self
    end
    alias_method :add_row, :append_row

    # Deletes logical row +index+ and its corresponding right-hand-side value.
    def delete_row(index)
      raise IndexError, "cannot delete from an empty window" if @root.nil?
      index = normalized_index(index)
      delete_at(index)
      numerical_check(:delete) unless @root.nil?
      self
    end

    # Replaces logical row +index+ and its right-hand-side value in one update.
    def replace_row(index, row, rhs)
      raise IndexError, "cannot replace a row in an empty window" if @root.nil?
      index = normalized_index(index)
      row = validated_row(row)
      rhs_values = validated_rhs_value(rhs)
      @root = replace_at(index, row, rhs_values)
      numerical_check(:replace)
      self
    end
    alias_method :update_row, :replace_row

    # Numerical rank of A computed from the most recent root QR factorization.
    def rank
      @root.nil? ? 0 : @root.rank
    end

    # Returns +true+ when the rank equals the number of columns.
    def full_rank?
      rank == @column_count
    end
    alias_method :full_column_rank?, :full_rank?

    # Returns the pivoted upper triangular factor at the root.
    def r
      return Matrix.empty(0, @column_count || 0) if @root.nil?
      @root.pivoted_r
    end
    alias_method :R, :r

    # Column permutation used by the rank-revealing factorization.
    def pivots
      @root.nil? ? Array.new(@column_count || 0) {|i| i } : @root.pivots.dup
    end
    alias_method :column_pivots, :pivots

    # Returns +true+ when Ax = rhs has an exact solution for every rhs column.
    def consistent?(tolerance = nil)
      return true if @root.nil?
      tolerance ||= residual_tolerance
      residual_squares.all? {|value| value <= tolerance * tolerance }
    end
    alias_method :solvable?, :consistent?

    # Returns the Euclidean residual norm for each right-hand side.
    def residual_norm
      residual_squares.map {|value| Math.sqrt(value) }
    end

    # Returns the minimum-norm least-squares solution.  A Vector is returned for
    # one right-hand side; otherwise a Matrix with one solution per column.
    def solve
      if (@column_count || 0).zero?
        return Vector.elements([], false) if @rhs_count == 1
        return Matrix.empty(0, @rhs_count)
      end
      raise Matrix::ErrDimensionMismatch, "the window has no right-hand side" if @rhs_count.zero?
      columns = solve_columns
      if @rhs_count == 1
        Vector.elements(columns.first, false)
      else
        Matrix.columns(columns)
      end
    end
    alias_method :minimum_norm_solution, :solve

    # Human-readable statement of rank, consistency, and the solution choice.
    def solution_status
      {
        row_count: row_count,
        column_count: @column_count || 0,
        rank: rank,
        full_rank: full_rank?,
        consistent: consistent?,
        residual_norm: residual_norm,
        solution_convention: solution_convention,
        rank_tolerance: effective_rank_tolerance
      }
    end

    # Explains the convention used by #solve.
    def solution_convention
      if full_rank? && row_count >= @column_count && @column_count && @column_count > 0
        UNIQUE_SOLUTION
      else
        MINIMUM_NORM_SOLUTION
      end
    end

    # Inspectable state used to decide whether a rebuild is necessary.
    def rebuild_policy
      {
        block_size: block_size,
        rank_tolerance: rank_tolerance,
        effective_rank_tolerance: effective_rank_tolerance,
        rebuild_tolerance: rebuild_tolerance,
        rebuild_count: rebuild_count,
        last_rebuild_reason: last_rebuild_reason,
        leaf_row_limits: leaf_row_limits,
        tree_height: @root.nil? ? 0 : @root.height,
        normal_equation_residual: normal_equation_residual
      }
    end

    # Forces all retained rows and right-hand-side values to be rebuilt.
    def rebuild!(reason = :manual)
      full_rebuild(reason)
      self
    end

    private

    def initialize_empty(column_count, rhs_count, options)
      options = options.is_a?(Hash) ? options : {}
      initialize_state(column_count, options[:block_size], options[:rank_tolerance],
                       options[:rebuild_tolerance])
      initialize_empty_rhs(rhs_count)
      self
    end

    def initialize_state(column_count, block_size, rank_tolerance, rebuild_tolerance)
      if column_count
        column_count = CoercionHelper.coerce_to_int(column_count)
        raise ArgumentError, "column_count cannot be negative" if column_count < 0
      end
      block_size = (block_size || DEFAULT_BLOCK_SIZE).to_int
      raise ArgumentError, "block_size must be positive" if block_size <= 0
      unless rebuild_tolerance.nil? || (rebuild_tolerance.is_a?(Numeric) && rebuild_tolerance > 0)
        raise ArgumentError, "rebuild_tolerance must be positive or nil"
      end
      unless rank_tolerance.nil? || (rank_tolerance.is_a?(Numeric) && rank_tolerance >= 0)
        raise ArgumentError, "rank_tolerance must be non-negative or nil"
      end

      @column_count = column_count
      @block_size = block_size
      @rank_tolerance = rank_tolerance
      @rebuild_tolerance = rebuild_tolerance || DEFAULT_REBUILD_TOLERANCE
      @rebuild_count = 0
      @last_rebuild_reason = nil
      @root = nil
      @row_count = 0
      @rhs_count = 0
      @normal = empty_matrix? ? nil : zero_square(column_count)
      @cross = nil
    end

    def initialize_rhs(rhs, rows)
      if rhs.nil?
        @rhs_count = 0
        @cross = nil
        return nil
      elsif rhs.is_a?(Matrix)
        raise Matrix::ErrDimensionMismatch unless rhs.row_count == rows
        values = rhs.to_a
        @rhs_count = rhs.column_count
        if @rhs_count.zero?
          @cross = nil
          return values
        end
      elsif rhs.is_a?(Vector)
        values = rhs.to_a.map {|value| [value] }
        raise Matrix::ErrDimensionMismatch unless values.size == rows
        @rhs_count = 1
      elsif rhs.is_a?(Numeric)
        raise Matrix::ErrDimensionMismatch unless rows == 1
        values = [[rhs]]
        @rhs_count = 1
      else
        converted = convert_to_array(rhs)
        raise Matrix::ErrDimensionMismatch unless converted.size == rows
        values = converted.map {|value| [value] }
        @rhs_count = 1
      end
      raise ArgumentError, "at least one right-hand side is required" if @rhs_count <= 0
      validate_real_matrix(values, "right-hand side")
      @cross = Array.new(@column_count) { Array.new(@rhs_count, 0.0) }
      values
    end

    def initialize_empty_rhs(rhs_count)
      rhs_count = rhs_count.nil? ? 0 : CoercionHelper.coerce_to_int(rhs_count)
      raise ArgumentError, "rhs_count cannot be negative" if rhs_count < 0
      @rhs_count = rhs_count
      @cross = (@column_count.nil? || rhs_count.zero?) ? nil :
                 Array.new(@column_count) { Array.new(rhs_count, 0.0) }
    end

    def empty_matrix?
      @column_count.nil? || @column_count.zero?
    end

    def zero_square(size)
      Array.new(size) { Array.new(size, 0.0) }
    end

    def validated_row(row)
      values = convert_to_array(row)
      if @column_count.nil?
        @column_count = values.size
        @normal = zero_square(@column_count)
        @cross = Array.new(@column_count) { Array.new(@rhs_count, 0.0) } unless @rhs_count.zero?
      elsif values.size != @column_count
        raise Matrix::ErrDimensionMismatch,
              "row has #{values.size} columns, expected #{@column_count}"
      end
      validate_real_matrix([values], "row")
      values.map(&:to_f)
    end

    def validated_rhs_value(rhs)
      values =
        if rhs.is_a?(Matrix)
          raise Matrix::ErrDimensionMismatch unless rhs.row_count == 1
          rhs.row(0).to_a
        elsif rhs.is_a?(Vector)
          rhs.to_a
        elsif rhs.is_a?(Numeric)
          [rhs]
        else
          convert_to_array(rhs)
        end

      if @rhs_count.zero?
        @rhs_count = values.size
        raise ArgumentError, "at least one right-hand side is required" if @rhs_count.zero?
        @cross = Array.new(@column_count) { Array.new(@rhs_count, 0.0) } unless @column_count.nil?
      elsif values.size != @rhs_count
        raise Matrix::ErrDimensionMismatch,
              "right-hand side has #{values.size} columns, expected #{@rhs_count}"
      end
      validate_real_matrix([values], "right-hand side")
      values.map(&:to_f)
    end

    def validate_real_matrix(rows, name)
      rows.each do |row|
        row.each do |value|
          unless value.is_a?(Numeric) && value.real? && finite_numeric?(value)
            raise ArgumentError, "#{name} contains a non-finite or non-real value"
          end
        end
      end
    end

    def finite_numeric?(value)
      value.respond_to?(:finite?) ? value.finite? : value.to_f.finite?
    end

    def normalized_index(index)
      index = CoercionHelper.coerce_to_int(index)
      index += @row_count if index < 0
      raise IndexError, "row #{index} is outside of 0...#{@row_count}" unless index >= 0 && index < @row_count
      index
    end

    def append_checked_row(row, rhs_values)
      leaf = new_leaf([[row, rhs_values]])
      @root = @root.nil? ? leaf : insert_leaf(@root, leaf)
      @row_count += 1
      update_normal(row, rhs_values, 1.0)
    end

    def delete_at(index, node = @root, offset = 0)
      local = index - offset
      if node.leaf
        row_pair = node.rows.delete_at(local)
        raise IndexError unless row_pair
        @row_count -= 1
        update_normal(row_pair[0], row_pair[1], -1.0)
        return node.rows.empty? ? nil : rebuild_leaf(node.rows)
      end

      left_rows = node.left.row_count
      if local < left_rows
        updated_left = delete_at(index, node.left, offset)
        node.left = updated_left
      else
        updated_right = delete_at(index, node.right, offset + left_rows)
        node.right = updated_right
      end
      return node.right if node.left.nil?
      return node.left if node.right.nil?
      balance(node)
    end

    def replace_at(index, row, rhs_values, node = @root, offset = 0)
      local = index - offset
      if node.leaf
        old_row, old_rhs = node.rows[local]
        raise IndexError unless old_row
        node.rows[local] = [row, rhs_values]
        update_normal(old_row, old_rhs, -1.0)
        update_normal(row, rhs_values, 1.0)
        return rebuild_leaf(node.rows)
      end

      left_rows = node.left.row_count
      if local < left_rows
        replace_at(index, row, rhs_values, node.left, offset)
      else
        replace_at(index, row, rhs_values, node.right, offset + left_rows)
      end
      balance(node)
      node
    end

    def insert_leaf(node, leaf)
      return leaf if node.nil?
      if node.leaf
        if node.rows.size < @block_size
          node.rows.concat(leaf.rows)
          return rebuild_leaf(node.rows)
        end
        return recombine(make_internal(node, leaf))
      end

      node.right = insert_leaf(node.right, leaf)
      balance(node)
    end

    def make_internal(left, right)
      node = Node.new
      node.left = left
      node.right = right
      node.leaf = false
      node
    end

    def balance(node)
      update_height(node)
      balance_factor = height(node.left) - height(node.right)
      if balance_factor > 1
        if node.left && height(node.left.left) < height(node.left.right)
          node.left = rotate_left(node.left)
        end
        node = rotate_right(node)
      elsif balance_factor < -1
        if node.right && height(node.right.right) < height(node.right.left)
          node.right = rotate_right(node.right)
        end
        node = rotate_left(node)
      end
      recombine(node)
      node
    end

    def rotate_right(node)
      pivot = node.left
      remaining_right = pivot.leaf ? node.right : combine_or_passthrough(pivot.right, node.right)
      new_left = pivot.leaf ? pivot : pivot.left
      combine_or_passthrough(new_left, remaining_right)
    end

    def rotate_left(node)
      pivot = node.right
      remaining_left = pivot.leaf ? node.left : combine_or_passthrough(node.left, pivot.left)
      new_right = pivot.leaf ? pivot : pivot.right
      combine_or_passthrough(remaining_left, new_right)
    end

    def combine_or_passthrough(left, right)
      return balance(left) if right.nil?
      return balance(right) if left.nil?
      balance(make_internal(left, right))
    end

    def height(node)
      node.nil? ? 0 : node.height
    end

    def update_height(node)
      node.height = 1 + [height(node.left), height(node.right)].max
    end

    def rebuild_leaf(row_pairs)
      new_leaf(row_pairs)
    end

    def new_leaf(row_pairs)
      rows = row_pairs.map(&:first)
      rhs_rows = row_pairs.map(&:last)
      factor = factor_rows(rows, rhs_rows)
      Node.new(row_pairs, nil, nil, nil, 1,
               factor[:r], factor[:pivoted_r], factor[:pivots], factor[:rank],
               factor[:pivoted_y], factor[:full_b], factor[:pivoted_full_b], rows.size, true)
    end

    def recombine(node)
      factor = combine_children(node.left, node.right)
      node.r = factor[:r]
      node.pivoted_r = factor[:pivoted_r]
      node.pivots = factor[:pivots]
      node.rank = factor[:rank]
      node.pivoted_y = factor[:pivoted_y]
      node.full_b = factor[:full_b]
      node.pivoted_full_b = factor[:pivoted_full_b]
      node.row_count = node.left.row_count + node.right.row_count
      node.leaf = false
      update_height(node)
      node
    end

    def combine_children(left, right)
      stacked_r = stack_full_rows(left, right, @column_count, :r)
      stacked_y = stack_full_rows(left, right, @rhs_count, :full_b)
      factor = factor_matrix(stacked_r, stacked_y, true)
      factor
    end

    def stack_full_rows(left, right, columns, field)
      left_rows = left.public_send(field).to_a
      right_rows = right.public_send(field).to_a
      if field == :full_b
        left_rows = left_rows.map {|row| row + Array.new(columns - row.size, 0.0) }
        right_rows = right_rows.map {|row| row + Array.new(columns - row.size, 0.0) }
      end
      left_rows + zero_rows(left.row_count - left_rows.size, columns) +
        right_rows + zero_rows(right.row_count - right_rows.size, columns)
    end

    def zero_rows(count, columns)
      Array.new([0, count].max) { Array.new(columns, 0.0) }
    end

    def factor_rows(rows, rhs_rows)
      factor_matrix(rows, rhs_rows, true)
    end

    def factor_matrix(rows, rhs_rows, pivot)
      m = rows.size
      n = @column_count || (rows.first || []).size
      k = @rhs_count
      unpivoted = qr_factorization(rows, rhs_rows, false, m, n, k, false)
      effective_rows = [m, n].min
      pivoted_stage = qr_factorization(unpivoted[:a].first(effective_rows),
                                       unpivoted[:b].first(effective_rows).map(&:dup),
                                       true, effective_rows, n, k, true)
      pivoted_a = pivoted_stage[:a] + unpivoted[:a].drop(effective_rows)
      pivoted_b = pivoted_stage[:b] + unpivoted[:b].drop(effective_rows)
      {
        r: matrix_from_rows(pad_rows(unpivoted[:a], n), n),
        pivoted_r: matrix_from_rows(pad_rows(pivoted_a, n), n),
        pivots: pivoted_stage[:pivots],
        rank: pivoted_stage[:rank],
        pivoted_y: matrix_from_rows(pivoted_stage[:b].first(effective_rows), k),
        full_b: matrix_from_rows(unpivoted[:b], k),
        pivoted_full_b: matrix_from_rows(pivoted_b, k)
      }
    end

    def padded_identity_product(q, m, n)
      return q if m >= n
      rows = q.to_a
      padded_rows = rows.map {|row| row + Array.new(n - m, 0.0) }
      (n - m).times do |offset|
        padded_rows << Array.new(n) {|column| column == m + offset ? 1.0 : 0.0 }
      end
      Matrix.send(:new, padded_rows, n)
    end

    def qr_factorization(rows, rhs_rows, use_pivot, m, n, k, rank_aware)
      a = rows.map(&:dup)
      b = rhs_rows.map {|row| row.dup }
      q = identity_matrix(m)
      pivots = Array.new(n) {|i| i }
      rank = 0
      column_norms = Array.new(n) {|j| sum_squares(rows, j) }

      [m, n].min.times do |iteration|
        pivot_index = iteration
        if use_pivot
          (iteration + 1 ... n).each do |candidate|
            pivot_index = candidate if column_norms[candidate] > column_norms[pivot_index]
          end
          swap_a_columns(a, iteration, pivot_index, pivots, column_norms)
        end

        norm_sq = column_norms[iteration]
        if !norm_sq.finite? || norm_sq <= 0
          break if rank_aware || norm_sq < 0
          update_remaining_norms(a, column_norms, iteration, m, n)
          next
        end
        scale = Math.sqrt(norm_sq)
        if rank_aware
          tolerance = rank_tolerance || default_rank_tolerance(scale, a, iteration)
          break if scale <= tolerance
        end

        vector = Array.new(m - iteration) {|i| a[iteration + i][iteration] }
        sign = vector.first < 0 ? -1.0 : 1.0
        vector[0] += sign * scale
        vector_norm_sq = vector.sum {|value| value * value }
        rank += 1

        if vector_norm_sq > 0
          beta = 2.0 / vector_norm_sq
          apply_householder(a, b, q, vector, beta, iteration, m, n, k)
        end
        a[iteration][iteration] = -sign * scale
        (iteration + 1 ... m).each {|i| a[i][iteration] = 0.0 }
        update_remaining_norms(a, column_norms, iteration, m, n)
      end

      {q: q, a: a, b: b, pivots: pivots, rank: rank}
    end

    def identity_matrix(size)
      Matrix.identity(size)
    end

    def sum_squares(rows, column)
      rows.sum(0.0) {|row| value = row[column] || 0.0; value * value }
    end

    def default_rank_tolerance(scale, a, iteration)
      global_scale = a.flatten.map(&:abs).max
      [a.size, a.first.size].max * Float::EPSILON * (global_scale.nonzero? || 1.0)
    end

    def swap_a_columns(a, left, right, pivots, norms)
      return if left == right
      a.each {|row| row[left], row[right] = row[right], row[left] }
      pivots[left], pivots[right] = pivots[right], pivots[left]
      norms[left], norms[right] = norms[right], norms[left]
    end

    def apply_householder(a, b, q, vector, beta, start, m, n, k)
      column_coefficients = (start ... n).map do |column|
        beta * (start ... m).sum(0.0) do |source_row|
          vector[source_row - start] * a[source_row][column]
        end
      end
      rhs_coefficients = Array.new(k) do |rhs_column|
        beta * (start ... m).sum(0.0) do |source_row|
          vector[source_row - start] * (b[source_row][rhs_column] || 0.0)
        end
      end
      (start ... m).each do |i|
        vi = vector[i - start]
        column_coefficients.each_with_index do |coefficient, offset|
          a[i][start + offset] -= vi * coefficient
        end
        next if k.zero?
        k.times do |rhs_column|
          current = b[i][rhs_column] || 0.0
          b[i][rhs_column] = current - vi * rhs_coefficients[rhs_column]
        end
      end

      qm = q.row_count
      (0 ... qm).each do |row|
        dot = 0.0
        (start ... m).each {|column| dot += q[row, column] * vector[column - start] }
        coefficient = beta * dot
        (start ... m).each do |column|
          q.send(:[]=, row, column, q[row, column] - coefficient * vector[column - start])
        end
      end
    end

    def update_remaining_norms(a, norms, iteration, m, n)
      (iteration + 1 ... n).each do |column|
        norms[column] = 0.0
        (iteration + 1 ... m).each do |row|
          value = a[row][column]
          norms[column] += value * value
        end
      end
    end

    def pad_rows(rows, columns)
      rows.map do |row|
        padded = row.dup
        padded.fill(0.0, padded.size ... columns)
        padded
      end
    end

    def matrix_from_rows(rows, columns)
      Matrix.send(:new, pad_rows(rows, columns), columns)
    end

    def column_squares(rows, columns = nil)
      columns ||= rows.first ? rows.first.size : 0
      Array.new(columns) do |column|
        rows.sum(0.0) {|row| value = row[column] || 0.0; value * value }
      end
    end

    def update_normal(row, rhs, sign)
      return if @column_count.nil?
      @column_count.times do |i|
        @column_count.times do |j|
          @normal[i][j] += sign * row[i] * row[j]
        end
        unless @rhs_count.zero?
          @rhs_count.times do |column|
            @cross[i][column] += sign * row[i] * rhs[column]
          end
        end
      end
    end

    def effective_rank_tolerance
      return rank_tolerance.to_f unless rank_tolerance.nil?
      root_scale = 1.0
      if @root && @root.pivoted_r.row_count.positive?
        root_scale = [root_scale, @root.pivoted_r.to_a.flatten.map(&:abs).max || 1.0].max
      end
      [row_count, @column_count || 0].max * Float::EPSILON * (root_scale.nonzero? || 1.0)
    end

    def residual_squares
      return Array.new(@rhs_count, 0.0) if @root.nil?
      column_squares(@root.full_b.to_a.drop(rank), @rhs_count)
    end

    def residual_tolerance
      rhs_scale = @rhs_count.times.map do |column|
        stored_rhs_column_square(column)
      end.map {|value| Math.sqrt([value, 0.0].max) }.max || 0.0
      [row_count, @column_count || 0, @rhs_count].max * Float::EPSILON *
        ((rhs_scale.nonzero? || 1.0))
    end

    def stored_rhs_column_square(column)
      sum = 0.0
      each_row_pair {|_, rhs| value = rhs[column]; sum += value * value }
      sum
    end

    def solve_columns
      n = @column_count || 0
      k = @rhs_count
      return Array.new(k) { [] } if n.zero?
      r = @root.nil? ? Matrix.empty(0, n) : @root.pivoted_r
      y = @root.nil? ? Matrix.empty(0, k) : @root.pivoted_y
      rank = self.rank
      pivots = self.pivots
      particular = Array.new(rank) { Array.new(k, 0.0) }
      transformed = Array.new(n) { Array.new(k, 0.0) }
      (rank - 1).downto(0) do |i|
        k.times do |column|
          value = y.row(i)[column]
          (i + 1 ... rank).each {|j| value -= r[i, j] * particular[j][column] }
          particular[i][column] = value / r[i, i]
        end
      end
      if rank == n
        transformed = particular
      else
        r12 = r.minor(0...rank, rank...n).to_a
        null_count = n - rank
        null_basis = Array.new(rank)
        (rank - 1).downto(0) do |i|
          null_basis[i] = Array.new(null_count) do |column|
            value = r12[i][column]
            (i + 1 ... rank).each {|j| value -= r[i, j] * null_basis[j][column] }
            value / r[i, i]
          end
        end
        null_values =
          if rank.zero?
            Array.new(null_count) { Array.new(k, 0.0) }
          elsif null_count <= rank
            ctc = Array.new(null_count) do |i|
              Array.new(null_count) do |j|
                value = i == j ? 1.0 : 0.0
                rank.times {|row| value += null_basis[row][i] * null_basis[row][j] }
                value
              end
            end
            right = Array.new(null_count) do |i|
              Array.new(k) do |column|
                Array.new(rank) {|row| null_basis[row][i] * particular[row][column] }.sum(0.0)
              end
            end
            solve_dense(ctc, right)
          else
            w_coefficients = Array.new(rank) do |i|
              Array.new(rank) do |j|
                value = i == j ? 1.0 : 0.0
                null_count.times {|column| value += null_basis[i][column] * null_basis[j][column] }
                value
              end
            end
            w_right = Array.new(rank) do |i|
              Array.new(k) do |column|
                Array.new(null_count) {|column_index| null_basis[i][column_index] * particular[i][column] }.sum(0.0)
              end
            end
            w = solve_dense(w_coefficients, w_right)
            Array.new(null_count) do |null_row|
              Array.new(k) do |column|
                Array.new(rank) {|row| null_basis[row][null_row] * w[row][column] }.sum(0.0)
              end
            end
          end
        rank.times do |i|
          k.times do |column|
            value = particular[i][column]
            null_count.times {|j| value -= null_basis[i][j] * null_values[j][column] }
            transformed[i][column] = value
          end
        end
        null_count.times {|i| k.times {|j| transformed[rank + i][j] = null_values[i][j] } }
      end

      solutions = Array.new(k) { Array.new(n, 0.0) }
      pivots.each_with_index do |original_column, transformed_column|
        k.times {|column| solutions[column][original_column] = transformed[transformed_column][column] }
      end
      solutions
    end

    def solve_dense(coefficients, right)
      size = coefficients.size
      factor = Matrix.send(:new, coefficients.map(&:dup), size).lup
      result = factor.solve(Matrix.send(:new, right.map(&:dup), right.first.size)).to_a
      result
    end

    def normal_equation_residual
      return 0.0 if @root.nil? || @rhs_count.zero? || (@column_count || 0).zero?
      pivoted_r = @root.pivoted_r
      pivots = @root.pivots
      rt_r = pivoted_r.transpose * pivoted_r
      full_b = @root.pivoted_full_b
      restored = Matrix.send(:new, Array.new(@column_count) do |i|
        Array.new(@column_count) do |j|
          transformed_i = pivots.index(i)
          transformed_j = pivots.index(j)
          transformed_i && transformed_j ? rt_r[transformed_i, transformed_j] : 0.0
        end
      end, @column_count)
      difference = restored - Matrix.send(:new, @normal, @column_count)
      scale = matrix_scale(restored).nonzero? || 1.0
      factor_error = matrix_scale(difference) / scale
      rhs_error = 0.0
      unless @cross.nil?
        active_rows = pivoted_r.row_count
        rt_y = pivoted_r.transpose * full_b.minor(0...active_rows, 0...@rhs_count)
        restored_cross = Matrix.send(:new, Array.new(@column_count) do |i|
          Array.new(@rhs_count) do |j|
            transformed_i = pivots.index(i)
            transformed_i ? rt_y[transformed_i, j] : 0.0
          end
        end, @rhs_count)
        rhs_difference = restored_cross - Matrix.send(:new, @cross, @rhs_count)
        rhs_scale = matrix_scale(rt_y).nonzero? || 1.0
        rhs_error = matrix_scale(rhs_difference) / rhs_scale
      end
      [factor_error, rhs_error].max
    rescue
      Float::INFINITY
    end

    def matrix_scale(matrix)
      values = []
      matrix.row_count.times do |i|
        matrix.column_count.times {|j| values << matrix[i, j].abs }
      end
      values.max || 0.0
    end

    def numerical_check(operation)
      return if @rebuild_tolerance.nil?
      error = normal_equation_residual
      if !error.finite? || error > @rebuild_tolerance
        full_rebuild("#{operation}:normal-equation-error")
      end
    end

    def full_rebuild(reason)
      pairs = []
      each_row_pair {|row, rhs| pairs << [row, rhs] }
      rhs_count = @rhs_count
      column_count = @column_count
      block_size = @block_size
      rank_tolerance = @rank_tolerance
      rebuild_tolerance = @rebuild_tolerance
      initialize_state(column_count, block_size, rank_tolerance, rebuild_tolerance)
      initialize_empty_rhs(rhs_count)
      pairs.each {|row, rhs| append_checked_row(row.dup, rhs.dup) }
      @rebuild_count += 1
      @last_rebuild_reason = reason.to_s
    end

    def each_row_pair(node = @root, &block)
      return if node.nil?
      if node.leaf
        node.rows.each(&block)
      else
        each_row_pair(node.left, &block)
        each_row_pair(node.right, &block)
      end
    end

    def leaf_row_limits(node = @root)
      return [] if node.nil?
      if node.leaf
        [node.rows.size]
      else
        leaf_row_limits(node.left) + leaf_row_limits(node.right)
      end
    end
  end

end
