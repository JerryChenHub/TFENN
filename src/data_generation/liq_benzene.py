import numpy as np
import csv
from opls2020.core.molecule import Benzene
from opls2020.core.force_field import OPLS2020_Force_Field

def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)

    x = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    y = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    z = np.sqrt(u1)     * np.sin(2 * np.pi * u3)
    w = np.sqrt(u1)     * np.cos(2 * np.pi * u3)

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    return R

def _min_interatomic_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    d = X1[:, None, :] - X2[None, :, :]
    return float(np.linalg.norm(d, axis=2).min())

def random_two_benzene_force_and_moment(rng=None,
                                        r_range: tuple[float, float] = (4.5, 8.0),
                                        smoothing: str = "linear",
                                        cutoff: float = 12.0,
                                        min_sep: float = 3.0,
                                        max_tries: int = 500
                                        ):
    """
    1) Sample R1, R2 ~ SO(3), and a displacement x with |x| in r_range (Å).
    2) Place molecule-1 at u1 = 0 with default orientation.
    3) Place molecule-2 at u2 = R1^T x, with v,w taken from columns 1 and 3 of R_rel = R1^T R2.
    4) Compute net force and net moment (about u1) acting on molecule-1 from molecule-2.

    Returns dict with R1, R2, x, u2, v, w, F_net, M_net, and per-atom forces on molecule-1.
    """
    if not rng:
        rng=np.random.default_rng()

    tries=0
    while True:
    # --- random scene ---
        R1 = _random_rotation_matrix(rng)
        R2 = _random_rotation_matrix(rng)

        # random x: direction ~ uniform on sphere, radius ~ Uniform[r_min, r_max]
        dir3 = rng.normal(size=3)
        dir3 /= np.linalg.norm(dir3)
        r = rng.uniform(*r_range)
        x = r * dir3

        # --- build molecules ---
        mol1 = Benzene()
        mol2 = Benzene()

        # mol1 at origin with default orientation (v=[1,0,0], w=[0,0,1])
        mol1.direction_vectors = (np.zeros(3), np.array([1., 0., 0.]), np.array([0., 0., 1.]))

        # mol2: u2 = R1^T x; (v,w) from columns of R_rel = R1^T R2
        u2 = R1.T @ x
        R_rel = R1.T @ R2
        v = R_rel[:, 0]   # first column
        w = R_rel[:, 2]   # third column
        mol2.direction_vectors = (u2, v, w)  # setter will re-orthonormalize & place atoms

        # --- pairwise non-bond forces on mol1 from mol2 ---
        X1 = mol1.atom_position  # (12,3)
        X2 = mol2.atom_position  # (12,3)

        if _min_interatomic_distance(X1, X2) >= min_sep:
            break
        if tries >= max_tries:
            raise RuntimeError(
                f"Could not sample a valid configuration with min_sep={min_sep} Å after {max_tries} attempts. "
                "Consider increasing r_range or reducing min_sep."
            )

    types1 = mol1._atom_types
    types2 = mol2._atom_types
    params = mol1.opls_params  # {"C":(...), "H":(...)} for benzene

    ff = OPLS2020_Force_Field(cutoff=cutoff, smoothing=smoothing)
    F_on_1 = np.zeros_like(X1)
    for i in range(len(X1)):
        p1 = params[types1[i]]
        for j in range(len(X2)):
            p2 = params[types2[j]]
            # Force on the FIRST atom (here: atom i in molecule-1)
            F_on_1[i] += ff.Non_bond_Force(X1[i], p1, X2[j], p2)

    # write back so net_force/net_moment use your class' definitions
    mol1._atom_force = F_on_1
    F_net = mol1.net_force     # (3,)
    M_net = mol1.net_moment    # (3,)

    return R_rel, R1.T@x, F_net, M_net


def generate_two_benzene_dataset(n_samples: int = 10_000,
                                 r_range: tuple[float, float] = (4.5, 8.0),
                                 smoothing: str = "linear",
                                 cutoff: float = 12.0,
                                 seed: int | None = None,
                                 min_sep: float = 3.0,
                                 max_tries: int = 500,
                                 gamma=1
                                 ) -> str:
    """
    Loop-sample and save a dataset with N rows:
      Columns = [R11..R13, R21..R23, R31..R33, x1,x2,x3, F1,F2,F3, M1,M2,M3]
      File    = train_2Benzene_{rmin}_{rmax}.cvs  (saved under `path` or current dir)

    Returns:
      The file path written.
    """
    rng = np.random.default_rng(seed)
    rmin, rmax = r_range
    fname = f"train_2Benzene_{n_samples}_{rmin}_{rmax}_{min_sep}_gamma{gamma}.cvs"


    header = [f"R{i}{j}" for i in range(1,4) for j in range(1,4)] \
           + ["x1","x2","x3"] \
           + ["F1","F2","F3"] \
           + ["M1","M2","M3"]

    with open("data/"+fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(n_samples):
            R_rel, x, F_net, M_net = random_two_benzene_force_and_moment(
                rng=rng, r_range=r_range, smoothing=smoothing, cutoff=cutoff,
                min_sep=min_sep, max_tries=max_tries
            )
            row = list(R_rel.reshape(-1)) + list(x) + list(F_net*gamma) + list(M_net*gamma)
            writer.writerow(row)

            # (optional) very light progress print every 1000 samples
            if (k+1) % 1000 == 0:
                print(f"[{k+1}/{n_samples}] written")


if __name__ == '__main__':

    # generate_two_benzene_dataset(n_samples=10000, r_range=(5.0, 10.0), min_sep=3.0,gamma=1)

    data = np.loadtxt("../../experiment1/data/train_2Benzene_10000_5.0_10.0_3.0_gamma1.cvs", delimiter=",", skiprows=1)
    F = data[:, 12:15]
    M = data[:, 15:18]
    F_norm = np.linalg.norm(F, axis=1)
    M_norm = np.linalg.norm(M, axis=1)
    F_mean, F_std, F_min, F_max, F_median = F_norm.mean(), F_norm.std(), F_norm.min(), F_norm.max(), np.median(F_norm)
    M_mean, M_std, M_max = M_norm.mean(), M_norm.std(), M_norm.max()
    print(f"F-norm -> mean: {F_mean:.6g}, std: {F_std:.6g}, min: {F_min:.6g}, max: {F_max:.6g}, median: {F_median:.6g}")
    print(f"M-norm -> mean: {M_mean:.6g}, std: {M_std:.6g}, max: {M_max:.6g}")