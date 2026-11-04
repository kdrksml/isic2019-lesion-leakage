# ── 6 · Losses and the global–local consistency term ──────────────────────────
class BaseLoss(nn.Module):
    def __init__(self, counts: Sequence[int]):
        super().__init__()
        n = torch.tensor(counts, dtype=torch.float32).clamp_min(1.0)
        self.register_buffer("counts", n)
        self.register_buffer("log_prior", torch.log(n / n.sum()))

    def set_epoch(self, epoch: int) -> None:
        pass

def _smooth_targets(y: torch.Tensor, n: int, eps: float) -> torch.Tensor:
    t = torch.full((y.size(0), n), eps / n, device=y.device)
    t.scatter_(1, y.unsqueeze(1), 1.0 - eps + eps / n)
    return t

class PlainCE(BaseLoss):                      # plain recipe
    def forward(self, z, y):
        return F.cross_entropy(z.float(), y)

class SmoothedCE(BaseLoss):                   # learning-rate grid
    def forward(self, z, y):
        return F.cross_entropy(z.float(), y, label_smoothing=CFG.LABEL_SMOOTH)

class FocalLogitAdjusted(BaseLoss):
    """Logit adjustment (Menon 2021) + label smoothing (Szegedy 2016) + gradient-carrying focal factor (Lin 2017)."""

    def forward(self, z, y):
        logp = F.log_softmax(z.float() + CFG.LA_TAU * self.log_prior, dim=1)
        ce = -(_smooth_targets(y, NC, CFG.LABEL_SMOOTH) * logp).sum(1)
        pt = logp.gather(1, y[:, None]).squeeze(1).exp()
        return ((1.0 - pt).pow(CFG.FOCAL_GAMMA) * ce).mean()

def make_loss(kind: str, counts: Sequence[int]) -> BaseLoss:
    return {"plain_ce": lambda: PlainCE(counts), "ce": lambda: SmoothedCE(counts),
            "focal_la": lambda: FocalLogitAdjusted(counts)}[kind]()

def supervised_term(loss: BaseLoss, out: Dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    return loss(out["logits"], y)

def consistency_term(loss: BaseLoss, z_global: torch.Tensor, z_local: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """GLC second-view term: supervised loss on the lesion-centred crop + one-directional distillation
    KL(p_global || p_local) with the global prediction as stop-gradient teacher (a partial view must agree with the
    full view, not the reverse; as in multi-crop self-distillation)."""
    lg, ll = F.log_softmax(z_global.float(), 1), F.log_softmax(z_local.float(), 1)
    kl = (lg.detach().exp() * (lg.detach() - ll)).sum(1).mean()          # gradient flows to the local view only
    return CFG.GLC_LOCAL_WEIGHT * loss(z_local, y) + CFG.GLC_CONSISTENCY_WEIGHT * kl

# sanity check on random logits: every loss is finite and differentiable
_c = [4522, 12875, 3323, 867, 2624, 239, 253, 628]
for _k in ("plain_ce", "ce", "focal_la"):
    _z = torch.randn(16, NC, requires_grad=True)
    _l = make_loss(_k, _c)(_z, torch.randint(0, NC, (16,)))
    _l.backward()
    assert torch.isfinite(_l) and torch.isfinite(_z.grad).all(), _k
print("losses OK")
