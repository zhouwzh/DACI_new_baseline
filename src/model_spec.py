"""Per-block LLM quantities: FLOPs, weight footprint, KV footprint, hidden-state sizes.

Per paper Table 2 / Sec 5.2.2:
    kappa_pf  = 2 * P * d^2 + P^2 * d        (FFN + attn)
    kappa_dec = 2 * d^2 + 4 * d * (P + t)    (FFN + attn over KV)
    z_pf  = 2 * P * d bytes (FP16)
    z_dec = 2 * d bytes (FP16)

Embedding anchored at N_0, LM head at a_S — not counted in per-block omega/chi/kappa.
"""
from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str
    L: int
    d: int
    omega_model_gb: float
    chi_unit_model_kb_per_token: float

    @property
    def omega_block_bytes(self) -> float:
        return self.omega_model_gb * (1024 ** 3) / self.L

    @property
    def chi_unit_block_bytes_per_token(self) -> float:
        return self.chi_unit_model_kb_per_token * 1024.0 / self.L

    def kappa_pf_per_block(self, P: int) -> float:
        return 2.0 * P * (self.d ** 2) + (P ** 2) * self.d

    def kappa_dec_per_block(self, P: int, t: int) -> float:
        return 2.0 * (self.d ** 2) + 4.0 * self.d * (P + t)

    def z_pf_bytes_with_P(self, P: int) -> float:
        return 2.0 * P * self.d

    def z_dec_bytes(self) -> float:
        return 2.0 * self.d

    def omega_stage_bytes(self, n_blocks: int) -> float:
        return n_blocks * self.omega_block_bytes

    def chi_stage_bytes(self, n_blocks: int, P: int, t: int) -> float:
        return n_blocks * (P + t) * self.chi_unit_block_bytes_per_token

    def kappa_pf_stage(self, n_blocks: int, P: int) -> float:
        return n_blocks * self.kappa_pf_per_block(P)

    def kappa_dec_stage(self, n_blocks: int, P: int, t: int) -> float:
        return n_blocks * self.kappa_dec_per_block(P, t)


def build_model_spec(name: str, cfg: dict) -> ModelSpec:
    m = cfg["models"][name]
    return ModelSpec(
        name=name, L=m["L"], d=m["d"],
        omega_model_gb=m["omega_model_gb"],
        chi_unit_model_kb_per_token=m["chi_unit_model_kb_per_token"],
    )


if __name__ == "__main__":
    import json
    with open("configs/models.json") as f:
        cfg = json.load(f)
    ms = build_model_spec("gemma-7b", cfg)
    print(f"{ms.name}: L={ms.L}, d={ms.d}")
    print(f"  kappa_pf(P=512) per block: {ms.kappa_pf_per_block(512)/1e9:.2f} GFLOPs")
    print(f"  kappa_dec(P=512, t=0) per block: {ms.kappa_dec_per_block(512, 0)/1e6:.2f} MFLOPs")
    print(f"  omega_block: {ms.omega_block_bytes/1e6:.1f} MB")
