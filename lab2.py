import numpy as np


class LPProblem:
    """
    Representation of a General Linear Programming Problem.
    Minimize or Maximize c^T x
    Subject to:
        Ax (op) b
        x_i bounds
    """

    def __init__(self, c: list,
                 A: list[list],
                 b: list,
                 ops: list[str],
                 bounds: list | None = None,
                 maximize: bool = False):
        """
        :param c: Coefficients of the objective function.
        :param A: Constraint matrix.
        :param b: Right-hand side of constraints.
        :param ops: List of strings ('<=', '>=', '=') for each constraint.
        :param bounds: List of tuples (lower, upper) for each variable.
                       None or (0, None) means x_i >= 0.
                       (None, None) means free variable.
        :param maximize: Boolean, True if objective is to maximize.
        """
        self.c: np.ndarray = np.array(c, dtype=float)
        self.A: np.ndarray = np.array(A, dtype=float)
        self.b: np.ndarray = np.array(b, dtype=float)
        self.ops = ops
        self.bounds = bounds if bounds is not None else [(0, None)] * len(c)
        self.maximize = maximize

    def __str__(self):
        res = ""
        direction = "Maximize" if self.maximize else "Minimize"
        res += f"{direction}: {self.c} * x\n"
        res += "Subject to:\n"
        for i in range(len(self.A)):
            res += f"  {np.around(self.A[i], 4)} {self.ops[i]} {np.around(self.b[i], 4)}\n"
        for i, bound in enumerate(self.bounds):
            if bound[0] is None and bound[1] is None:
                res += f"x_{i+1} <> 0"
            elif bound[0] is None:
                res += f"x_{i+1} <= {bound[1]}\n"
            elif bound[1] is None:
                res += f"x_{i+1} >= {bound[0]}\n"
            else:
                res += f"x_{i+1} >= {bound[0]}\n"
                res += f"x_{i+1} <= {bound[1]}\n"

        return res

    def canonicalize(self):
        """
        Converts the LP to Canonical Form (Canonicalization):
        Maximize c^T x
        Subject to:
            Ax = b
            x >= 0
        """
        c_new = self.c.copy() if self.maximize else -self.c.copy()

        # 2. Variable transformations (ensure all x_i >= 0)
        transformed_c = []
        transformed_A_cols = []
        obj_constant = 0.0
        rhs = self.b.copy()

        finite_upper_bounds = []

        current_new_idx = 0
        for i in range(len(self.c)):
            l, u = self.bounds[i]
            if l == 0 and u is None:
                transformed_c.append(c_new[i])
                transformed_A_cols.append(self.A[:, i])
                current_new_idx += 1
            elif l is None and u is None:
                transformed_c.extend([c_new[i], -c_new[i]])
                transformed_A_cols.extend([self.A[:, i], -self.A[:, i]])
                current_new_idx += 2
            elif l is not None and u is None:
                transformed_c.append(c_new[i])
                transformed_A_cols.append(self.A[:, i])
                rhs -= self.A[:, i] * l
                obj_constant += c_new[i] * l
                current_new_idx += 1
            elif l is None and u is not None:
                transformed_c.append(-c_new[i])
                transformed_A_cols.append(-self.A[:, i])
                rhs -= self.A[:, i] * u
                obj_constant += c_new[i] * u
                current_new_idx += 1
            else:
                transformed_c.append(c_new[i])
                transformed_A_cols.append(self.A[:, i])
                rhs -= self.A[:, i] * l
                obj_constant += c_new[i] * l
                finite_upper_bounds.append((current_new_idx, u - l))
                current_new_idx += 1

        A_norm = np.column_stack(transformed_A_cols)
        c_norm = np.array(transformed_c)
        b_norm = rhs

        for idx, limit in finite_upper_bounds:
            new_row = np.zeros(A_norm.shape[1])
            new_row[idx] = 1.0
            A_norm = np.vstack([A_norm, new_row])
            b_norm = np.append(b_norm, limit)

        all_ops = list(self.ops) + ['<='] * len(finite_upper_bounds)

        final_A = A_norm
        final_c = c_norm
        final_b = b_norm

        num_constraints = len(all_ops)
        slack_cols = []

        for i, op in enumerate(all_ops):
            if op == '<=':
                col = np.zeros(num_constraints)
                col[i] = 1.0
                slack_cols.append(col)
                final_c = np.append(final_c, 0.0)
            elif op == '>=':
                col = np.zeros(num_constraints)
                col[i] = -1.0
                slack_cols.append(col)
                final_c = np.append(final_c, 0.0)

        if slack_cols:
            padding = np.column_stack(slack_cols)
            final_A = np.column_stack([final_A, padding])

        for i in range(len(final_b)):
            if final_b[i] < 0:
                final_A[i] = -final_A[i]
                final_b[i] = -final_b[i]

        return final_c, final_A, final_b, obj_constant

    def normalize(self):
        """
        Converts the LP to Normal Form:
        Maximize c^T x
        Subject to:
            Ax <= b
            x >= 0
        """
        c_new = self.c.copy() if self.maximize else -self.c.copy()

        transformed_c = []
        transformed_A_cols = []
        obj_constant = 0.0
        rhs = self.b.copy()
        finite_upper_bounds = []

        current_new_idx = 0
        for i in range(len(self.c)):
            l, u = self.bounds[i]
            if l == 0 and u is None:
                transformed_c.append(c_new[i])
                transformed_A_cols.append(self.A[:, i])
                current_new_idx += 1
            elif l is None and u is None:
                transformed_c.extend([c_new[i], -c_new[i]])
                transformed_A_cols.extend([self.A[:, i], -self.A[:, i]])
                current_new_idx += 2
            elif l is not None and u is None:
                transformed_c.append(c_new[i])
                transformed_A_cols.append(self.A[:, i])
                rhs -= self.A[:, i] * l
                obj_constant += c_new[i] * l
                current_new_idx += 1
            elif l is None and u is not None:
                transformed_c.append(-c_new[i])
                transformed_A_cols.append(-self.A[:, i])
                rhs -= self.A[:, i] * u
                obj_constant += c_new[i] * u
                current_new_idx += 1
            else:
                transformed_c.append(c_new[i])
                transformed_A_cols.append(self.A[:, i])
                rhs -= self.A[:, i] * l
                obj_constant += c_new[i] * l
                finite_upper_bounds.append((current_new_idx, u - l))
                current_new_idx += 1

        A_can = np.column_stack(transformed_A_cols)
        c_can = np.array(transformed_c)
        b_can = rhs

        final_A = []
        final_b = []

        for i, op in enumerate(self.ops):
            if op == '<=':
                final_A.append(A_can[i])
                final_b.append(b_can[i])
            elif op == '>=':
                final_A.append(-A_can[i])
                final_b.append(-b_can[i])
            elif op == '=':
                final_A.append(A_can[i])
                final_b.append(b_can[i])
                final_A.append(-A_can[i])
                final_b.append(-b_can[i])

        for idx, limit in finite_upper_bounds:
            row = np.zeros(A_can.shape[1])
            row[idx] = 1.0
            final_A.append(row)
            final_b.append(limit)

        return c_can, np.array(final_A), np.array(final_b), obj_constant


def print_lp(c, A, b, offset=0, mode="Normal"):
    print(f"\n--- {mode} Form ---")
    print(f"Maximize: {c} * x + {offset}")
    print("Subject to:")
    op_str = "<=" if mode == "Normal" else "="
    for i in range(len(A)):
        print(f"  {np.around(A[i], 4)} {op_str} {np.around(b[i], 4)}")
    print("x >= 0")


if __name__ == "__main__":
    c = [1, 1, 0, 0, 0]
    A = [[-1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]]
    b = [1, 3, 2]
    ops = ['=', '=', '=']
    bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]

    lp = LPProblem(c, A, b, ops, bounds, maximize=False)

    c_std, A_std, b_std, off_std = lp.normalize()
    print_lp(c_std, A_std, b_std, off_std, "Normal")

    c_can, A_can, b_can, off_can = lp.canonicalize()
    print_lp(c_can, A_can, b_can, off_can, "Canonical")
