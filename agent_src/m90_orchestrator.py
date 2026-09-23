from __future__ import annotations  # bundle:strip
from agent_src.contract import *  # noqa: F401,F403  bundle:strip
try:  # bundle:strip
    from agent_src.m10_dataview import DataView, load_history, load_tariff_descriptions  # noqa: F401  bundle:strip
except ImportError:  # bundle:strip
    pass  # bundle:strip
try:  # bundle:strip
    from agent_src.m20_prior import PriorBuilder  # noqa: F401  bundle:strip
except ImportError:  # bundle:strip
    pass  # bundle:strip
try:  # bundle:strip
    from agent_src.m30_arm_model import ArmModel  # noqa: F401  bundle:strip
except ImportError:  # bundle:strip
    pass  # bundle:strip
try:  # bundle:strip
    from agent_src.m40_simulator import ScoreSimulator  # noqa: F401  bundle:strip
except ImportError:  # bundle:strip
    pass  # bundle:strip
try:  # bundle:strip
    from agent_src.m50_allocator import Allocator  # noqa: F401  bundle:strip
except ImportError:  # bundle:strip
    pass  # bundle:strip
try:  # bundle:strip
    from agent_src.m60_packer import CampaignPacker  # noqa: F401  bundle:strip
except ImportError:  # bundle:strip
    pass  # bundle:strip
try:  # bundle:strip
    from agent_src.m70_planner import PilotPlanner  # noqa: F401  bundle:strip
except ImportError:  # bundle:strip
    pass  # bundle:strip
try:  # bundle:strip
    from agent_src.m80_llm import LLMLayer  # noqa: F401  bundle:strip
except ImportError:  # bundle:strip
    pass  # bundle:strip
try:  # bundle:strip
    from agent_src.m85_guardrails import Guardrails, Reporter  # noqa: F401  bundle:strip
except ImportError:  # bundle:strip
    pass  # bundle:strip

import asyncio
import math
import os
import threading
import time
from typing import Any, Callable, Optional

import pandas as pd

# Orchestrator: async pipeline research -> hypotheses -> explore -> allocate -> review -> finalize.
# Оркестратор: асинхронный конвейер этапов с общим дедлайном и безопасными fallback-ами.

_ORC_COMPONENT_NAMES: tuple[str, ...] = (
    "DataView",
    "load_history",
    "load_tariff_descriptions",
    "PriorBuilder",
    "ArmModel",
    "ScoreSimulator",
    "Allocator",
    "CampaignPacker",
    "PilotPlanner",
    "LLMLayer",
    "Guardrails",
    "Reporter",
)
_ORC_MAX_CONSECUTIVE_FAILURES = 3
_ORC_CALIBRATE_PRIOR = True  # re-fit prior bias / extra variance from pilots after every result
_ORC_VERIFY_PILOTS = 8  # pilots reserved for verifying the arms the coarse plan would deploy
_ORC_VERIFY_MONEY_FRAC = 0.1  # pilot on the deploy channel only if it costs <= this share of the expected loss
_ORC_COARSE_Z_FRAC = 0.5  # coarse packing ranks arms by mean − 0.5·z_risk·sd
_ORC_FILTER_KEYS = ("filter_arpu_segment", "filter_data_segment", "filter_call_segment", "filter_current_tariff")


def _orc_base_dir() -> str:
    """Directory holding agent.py (repo root when running from the agent_src package)."""
    try:
        d = os.path.dirname(os.path.abspath(__file__))
    except NameError:  # pragma: no cover - exotic loaders
        return os.getcwd()
    if os.path.basename(d) == "agent_src":
        d = os.path.dirname(d)
    return d


def _orc_search_dirs() -> list[str]:
    """Deduplicated dirs to look for data files: agent dir, then cwd."""
    out: list[str] = []
    for d in (_orc_base_dir(), os.getcwd()):
        if d and d not in out:
            out.append(d)
    return out


def _orc_load_dotenv() -> None:
    """Load <agent dir>/.env (and cwd .env) via python-dotenv if installed; never overrides env."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    for d in _orc_search_dirs():
        path = os.path.join(d, ".env")
        try:
            if os.path.isfile(path):
                load_dotenv(path, override=False)
        except Exception:
            continue


def _orc_is_missing(v: Any) -> bool:
    """None / NaN / pd.NA / empty string -> True (filter not set; mirrors organizer `pd.notna`)."""
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _orc_campaign_matches(camp: dict, sub: tuple) -> bool:
    """True if a campaign's filters select sub-cell `sub` (tariff, arpu, data, call)."""
    cur, arpu, data, call = sub
    for key, val in (("filter_arpu_segment", arpu), ("filter_data_segment", data), ("filter_call_segment", call)):
        f = camp.get(key)
        if not _orc_is_missing(f) and str(f).strip() != val:
            return False
    f = camp.get("filter_current_tariff")
    if not _orc_is_missing(f):
        wanted = {t.strip() for t in str(f).split(";") if t.strip()}
        if cur not in wanted:
            return False
    return True


def _orc_option_in_campaign(camp: dict, opt: Any) -> bool:
    """Option belongs to campaign: same target/channel and filters cover its sub."""
    return (
        camp.get("target_tariff") == opt.target
        and camp.get("channel") == opt.channel
        and _orc_campaign_matches(camp, tuple(opt.sub))
    )


def _orc_emergency_campaigns(env: Any) -> list[dict]:
    """Last resort without any module: one valid push campaign whose filters match nobody, else []."""
    try:
        tariffs = [str(t) for t in list(env.tariffs["tariff_plan_code"])]
        channels = list(env.channels)
        if not tariffs or not channels:
            return []
        channel = "push" if "push" in channels else sorted(channels)[0]
        prof = env.customer_profile
        seen = set(
            zip(
                prof["current_tariff"].astype(str),
                prof["arpu_segment"].astype(str),
                prof["data_segment"].astype(str),
                prof["call_segment"].astype(str),
            )
        )
        for cur in tariffs:
            for a in ARPU_SEGMENTS:
                for d in DATA_SEGMENTS:
                    for c in CALL_SEGMENTS:
                        if (cur, a, d, c) in seen:
                            continue
                        target = next((t for t in tariffs if t != cur), None)
                        if target is None:
                            return []
                        return [
                            campaign_dict(
                                campaign_name="c00_noop",
                                filter_arpu_segment=a,
                                filter_data_segment=d,
                                filter_call_segment=c,
                                filter_current_tariff=cur,
                                target_tariff=target,
                                channel=channel,
                            )
                        ]
    except Exception:
        return []
    return []


def _orc_finite(x: Any) -> bool:
    """True for a real finite number."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _orc_weight_key(k: Any) -> tuple:
    """LLM weight key -> ArmKey (accepts tuples/lists or "from|seg|to" strings)."""
    if isinstance(k, str):
        return parse_arm_id(k)
    return tuple(k)


def run_coro(coro: Any) -> Any:
    """Run a coroutine to completion: asyncio.run, or a dedicated thread + loop if a loop is running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # propagated to the caller thread
            box["error"] = exc

    th = threading.Thread(target=_runner, name="agent-orchestrator", daemon=True)
    th.start()
    th.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


class Orchestrator:
    """Async pipeline driver; every stage is guarded and the run always yields campaigns."""

    def __init__(
        self,
        cfg: Config,
        components: Optional[dict[str, Any]] = None,
        llm_mode: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.components = dict(components or {})
        self.llm_mode = llm_mode
        self.log = RunLog()
        self.stages: dict[str, str] = {}
        self.dv: Any = None
        self.history: Any = None
        self.descriptions: dict[str, str] = {}
        self.priors: dict = {}
        self.model: Any = None
        self.sim: Any = None
        self.allocator: Any = None
        self.planner: Any = None
        self.packer: Any = None
        self.llm: Any = None
        self.plan: Any = None
        self._verify_packer: Any = None
        self._verify_skip: set = set()
        self.campaigns: list[dict] = []
        self.weights: dict = {}
        self.review: Any = None
        self.explore_money_spent = 0.0
        self.explore_reach_spent = 0
        self.pilots_per_arm: dict = {}
        self.used_subs: dict = {}
        self.pilot_records: dict = {}  # SubKey -> list[(ArmKey, channel, n)]
        self.extra: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers

    def _c(self, name: str) -> Any:
        """Resolve a component (injected override, else module-level name)."""
        if name in self.components:
            return self.components[name]
        obj = globals().get(name)
        if obj is None:
            raise NameError(f"component {name} unavailable")
        return obj

    def _fail(self, stage: str, exc: BaseException) -> None:
        self.stages[stage] = f"error: {type(exc).__name__}"
        self.log.log("stage_error", stage=stage, error=f"{type(exc).__name__}: {str(exc)[:300]}")

    def _ratio_fn(self) -> Callable[[str, str, str, str], float]:
        """Posterior-mean lift ratio per (tariff, arpu, target, channel), memoised."""
        cache: dict[tuple, float] = {}
        model = self.model

        def fn(cur: str, arpu: str, target: str, channel: str) -> float:
            key = (cur, arpu, target, channel)
            if key not in cache:
                try:
                    v = float(model.posterior((cur, arpu, target), channel).mean) if model is not None else 0.0
                except Exception:
                    v = 0.0
                cache[key] = v if math.isfinite(v) else 0.0
            return cache[key]

        return fn

    def _simulate(self, campaigns: list[dict], budget: float, contacts: int) -> Any:
        return self.sim.simulate(campaigns, self._ratio_fn(), budget, contacts)

    # ------------------------------------------------------------------ run

    async def run(self, env: Any) -> list[dict]:
        """Execute all stages under the global time budget; never raises (except cancellation)."""
        cfg = self.cfg
        t0 = time.monotonic()
        hard_deadline = t0 + float(cfg.time_budget_s)
        reserve = min(0.25 * float(cfg.time_budget_s), 90.0)
        learn_deadline = hard_deadline - reserve
        mode = self.llm_mode if self.llm_mode in LLM_MODES else resolve_llm_mode()
        self.extra["llm_mode"] = mode
        self.log.log("start", llm_mode=mode, time_budget_s=cfg.time_budget_s)

        async def _learn() -> None:
            await self._stage_research(env)
            await self._stage_hypotheses(mode, learn_deadline)
            await self._stage_explore(env, learn_deadline)

        async def _post() -> None:
            await self._stage_allocate(env)
            await self._stage_review(env, mode, hard_deadline)

        try:
            await asyncio.wait_for(_learn(), timeout=max(learn_deadline - time.monotonic(), 0.0))
        except (TimeoutError, asyncio.TimeoutError):
            self.log.log("timeout", stage="learn", pilots=len(self.log.pilots))
            self.stages.setdefault("timeout", "learn")
        except Exception as exc:  # defensive: stages already guard themselves
            self._fail("learn", exc)

        if self.dv is None:
            self.log.log("no_dataview")
            return self._finalize(env, [])

        try:
            post_left = max(hard_deadline - time.monotonic(), 20.0)
            await asyncio.wait_for(_post(), timeout=post_left)
        except (TimeoutError, asyncio.TimeoutError):
            self.log.log("timeout", stage="post")
            self.stages.setdefault("timeout", "post")
        except Exception as exc:
            self._fail("post", exc)
        return self._finalize(env, self.campaigns)

    # ------------------------------------------------------------------ stages

    async def _stage_research(self, env: Any) -> None:
        """DataView + history in parallel threads, then priors."""
        try:
            DataView_ = self._c("DataView")
            dirs = _orc_search_dirs()

            async def _hist() -> Any:
                try:
                    return await asyncio.to_thread(self._c("load_history"), dirs)
                except Exception as exc:
                    self._fail("history", exc)
                    return None

            async def _desc() -> dict:
                try:
                    return await asyncio.to_thread(self._c("load_tariff_descriptions"), dirs) or {}
                except Exception as exc:
                    self._fail("descriptions", exc)
                    return {}

            dv, hist, desc = await asyncio.gather(
                asyncio.to_thread(DataView_, env.customer_profile, env.tariffs, env.channels), _hist(), _desc()
            )
            self.dv, self.history, self.descriptions = dv, hist, dict(desc)
            self.log.log(
                "research",
                cells=len(getattr(dv, "cells", {}) or {}),
                subs=len(getattr(dv, "subs", {}) or {}),
                history=None if hist is None else int(len(hist)),
            )
        except Exception as exc:
            self._fail("research", exc)
            return
        try:
            self.priors = await asyncio.to_thread(self._c("PriorBuilder")(self.cfg).build, self.history, self.dv) or {}
            self.stages["research"] = "ok"
        except Exception as exc:
            self.priors = {}
            self._fail("priors", exc)
        try:
            self.sim = self._c("ScoreSimulator")(self.dv, env.customer_profile)
        except Exception as exc:
            self._fail("simulator", exc)
        try:
            self.model = self._c("ArmModel")(self.cfg, self.dv, self.priors)
            self.allocator = self._c("Allocator")(self.cfg, self.dv, self.model)
            self.planner = self._c("PilotPlanner")(self.cfg, self.dv, self.model, self.allocator)
        except Exception as exc:
            self._fail("engine", exc)

    def _hypothesis_batches(self) -> dict[str, list[dict]]:
        """Top arms by ΣP_cell·(mu+sd), grouped by ARPU segment."""
        scored: list[tuple[float, tuple, Any]] = []
        cells = getattr(self.dv, "cells", {}) or {}
        for arm in sorted(self.priors):
            pr = self.priors[arm]
            cell = cells.get((arm[0], arm[1]))
            if cell is None:
                continue
            s = float(cell.sum_p) * (float(pr.mu) + float(pr.sd))
            scored.append((-s, tuple(arm), pr))
        scored.sort(key=lambda t: (t[0], t[1]))
        batches: dict[str, list[dict]] = {}
        for _, arm, pr in scored[: int(self.cfg.top_arms)]:
            cell = cells[(arm[0], arm[1])]
            batches.setdefault(arm[1], []).append(
                {
                    "arm_id": arm_id(arm),
                    "from_tariff": arm[0],
                    "arpu_segment": arm[1],
                    "target_tariff": arm[2],
                    "prior_mu": float(pr.mu),
                    "prior_sd": float(pr.sd),
                    "prior_share": float(pr.share),
                    "n_hist": int(pr.n_hist),
                    "prior_source": str(pr.source),
                    "cell_n": int(cell.n),
                    "cell_sum_p": float(cell.sum_p),
                }
            )
        return {k: batches[k] for k in sorted(batches)}

    def _tariff_table(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for code in list(getattr(self.dv, "tariff_codes", []) or []):
            try:
                info = dict(self.dv.tariff_info(code))
            except Exception:
                info = {}
            if code in self.descriptions:
                info["description"] = self.descriptions[code]
            out[code] = info
        return out

    async def _stage_hypotheses(self, mode: str, learn_deadline: float) -> None:
        """LLM plausibility weights for the top arms (neutral on any failure)."""
        if self.dv is None or not self.priors:
            self.stages["hypotheses"] = "skipped"
            return
        try:
            cache_path = os.path.join(_orc_base_dir(), self.cfg.cache_file)
            self.llm = self._c("LLMLayer")(self.cfg, mode, cache_path, self.log)
            batches = self._hypothesis_batches()
            budget = min(float(self.cfg.llm_budget_s), max(learn_deadline - time.monotonic() - 5.0, 0.0))
            if not batches or budget <= 0:
                self.stages["hypotheses"] = "skipped"
                return
            w = await asyncio.wait_for(self.llm.assess_hypotheses(batches, self._tariff_table()), timeout=budget)
            self.weights = {}
            for k, v in sorted(dict(w or {}).items(), key=lambda kv: str(kv[0])):
                try:
                    key, val = _orc_weight_key(k), float(v)
                except (TypeError, ValueError):
                    continue
                if _orc_finite(val):
                    self.weights[key] = val
            if self.planner is not None and self.weights:
                self.planner.set_weights(self.weights)
            n_up = sum(1 for v in self.weights.values() if v > 1.0)
            n_down = sum(1 for v in self.weights.values() if v < 1.0)
            self.log.log("hypotheses", arms=len(self.weights), up=n_up, down=n_down)
            self.stages["hypotheses"] = "ok"
        except (TimeoutError, asyncio.TimeoutError) as exc:
            if time.monotonic() >= learn_deadline:
                raise
            self._fail("hypotheses", exc)
        except Exception as exc:
            self._fail("hypotheses", exc)

    def _explore_state(self, env: Any, deadline: float) -> ExploreState:
        return ExploreState(
            remaining_budget=float(env.remaining_budget),
            remaining_contacts=int(env.remaining_contacts),
            pilots_left=int(env.pilots_left),
            explore_money_spent=float(self.explore_money_spent),
            explore_reach_spent=int(self.explore_reach_spent),
            pilots_per_arm=dict(self.pilots_per_arm),
            used_subs=dict(self.used_subs),
            deadline=deadline,
        )

    async def _stage_explore(self, env: Any, deadline: float) -> None:
        """Sequential pilots chosen by the planner; results feed the arm model."""
        if self.planner is None or self.model is None:
            self.stages["explore"] = "skipped"
            return
        failures = 0
        max_iter = int(self.cfg.max_pilots) + 2 * _ORC_MAX_CONSECUTIVE_FAILURES + 5
        self.stages["explore"] = "ok"
        can_verify = self.sim is not None and hasattr(self._c("CampaignPacker"), "pack_coarse")
        reserve = min(_ORC_VERIFY_PILOTS, int(env.pilots_left)) if can_verify else 0
        phase = "kg"
        for _ in range(max_iter):
            if int(env.pilots_left) <= 0 or time.monotonic() >= deadline:
                break
            spec = None
            if phase == "kg":
                if int(env.pilots_left) <= reserve:
                    phase = "verify"
                else:
                    state = self._explore_state(env, deadline)
                    state.pilots_left = max(int(state.pilots_left) - reserve, 0)
                    try:
                        spec = await asyncio.to_thread(self.planner.next_pilot, state)
                    except Exception as exc:
                        self._fail("explore", exc)
                        spec = None
                    if spec is None:
                        self.log.log("explore_stop", reason="planner", pilots=len(self.log.pilots))
                        phase = "verify"
            if spec is None and phase == "verify":
                if not can_verify:
                    break
                try:
                    spec = await asyncio.to_thread(self._verify_spec, env)
                except Exception as exc:
                    self._fail("verify", exc)
                    break
                if spec is None:
                    self.log.log("verify_stop", pilots=len(self.log.pilots))
                    break
            kw = spec.run_kwargs()
            pre = self._env_counters(env)
            try:
                result = env.run_pilot(**kw)
                if not isinstance(result, dict):
                    raise ValueError("run_pilot returned non-dict")
                y = float(result["observed_lift_ratio"])
                n = int(result["n_customers"])
                cost = float(result["cost"])
                if not (_orc_finite(y) and _orc_finite(cost)) or n <= 0 or cost < 0:
                    raise ValueError(f"malformed pilot result y={y} n={n} cost={cost}")
            except Exception as exc:  # RuntimeError/ValueError from env, or malformed result
                self._account_env_delta(env, pre, spec)
                failures += 1
                self.log.log("pilot_error", arm=arm_id(spec.arm), channel=spec.channel,
                             error=f"{type(exc).__name__}: {str(exc)[:200]}")
                self.log.pilots.append({"arm": arm_id(spec.arm), "channel": spec.channel, "n_req": int(spec.n),
                                        "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                try:
                    self.planner.register_result(spec, {"error": str(exc)})
                except Exception as exc2:
                    self._fail("planner_register", exc2)
                if failures >= _ORC_MAX_CONSECUTIVE_FAILURES:
                    break
                continue
            failures = 0
            obs = Observation(arm=tuple(spec.arm), channel=spec.channel, y=y, n=n, cost=cost,
                              sub=None if spec.sub is None else tuple(spec.sub),
                              pilot_index=max(len(getattr(env, "pilot_history", []) or []) - 1, 0))
            self.explore_money_spent += cost
            self.explore_reach_spent += n
            arm = tuple(spec.arm)
            self.pilots_per_arm[arm] = self.pilots_per_arm.get(arm, 0) + 1
            if spec.sub is not None:
                sub = tuple(spec.sub)
                self.used_subs[sub] = self.used_subs.get(sub, 0) + n
                self.pilot_records.setdefault(sub, []).append((arm, str(spec.channel), n))
            try:
                self.model.add(obs)
            except Exception as exc:
                self._fail("model_add", exc)
            if _ORC_CALIBRATE_PRIOR and hasattr(self.model, "calibrate"):
                try:
                    cal = self.model.calibrate()
                    self.extra["calibration"] = cal
                except Exception as exc:
                    self._fail("calibrate", exc)
            try:
                self.planner.register_result(spec, result)
            except Exception as exc:
                self._fail("planner_register", exc)
            self.log.pilots.append({
                "index": obs.pilot_index, "arm": arm_id(arm), "channel": spec.channel, "n_req": int(spec.n),
                "n": n, "cost": cost, "y": y, "sub": None if spec.sub is None else "|".join(spec.sub),
                "score": float(spec.score) if spec.score is not None else None, "reason": str(spec.reason)[:200],
            })
            self.log.log("pilot", arm=arm_id(arm), channel=spec.channel, n=n, y=round(y, 4), cost=cost)
            await asyncio.sleep(0)  # cancellation point for the global deadline

    def _verify_spec(self, env: Any) -> Optional[PilotSpec]:
        """Verification pilot: the deployed arm with the largest expected downside in the current coarse plan.

        The coarse plan (as it would be deployed now) is built; for every deployed (arm, channel) the expected
        loss of deploying it is sum_p·E[max(0, −r)] under the posterior. The riskiest arm is piloted (freshest
        sub of its cell, n = max_pilot) if that loss exceeds the pilot's money plus the contacts' opportunity value
        (half the plan's average net per contact; zero when the plan leaves enough reach unused).
        """
        cfg = self.cfg
        budget, contacts = float(env.remaining_budget), int(env.remaining_contacts)
        if self._verify_packer is None:
            self._verify_packer = self._c("CampaignPacker")(cfg, self.dv, self.sim)
        packer = self._verify_packer
        camps = packer.pack_coarse(self.model, budget, contacts, z=_ORC_COARSE_Z_FRAC * float(cfg.z_risk))
        if not camps:
            return None
        sim = packer.last_sim
        used = int(getattr(sim, "contacts", 0) or 0) if sim is not None else 0
        v_contact = 0.5 * max(float(getattr(sim, "net", 0.0) or 0.0), 0.0) / used if used > 0 else 0.0
        cells = getattr(self.dv, "cells", {}) or {}
        stakes: dict[tuple, float] = {}
        for camp in camps:
            tg, ch, a = str(camp.get("target_tariff")), str(camp.get("channel")), str(camp.get("filter_arpu_segment"))
            for t in [x.strip() for x in str(camp.get("filter_current_tariff") or "").split(";") if x.strip()]:
                cell = cells.get((t, a))
                if cell is not None and t != tg:
                    key = ((t, a, tg), ch)
                    stakes[key] = stakes.get(key, 0.0) + float(cell.sum_p)
        cap = int(cfg.max_pilots_per_arm)
        best: Optional[tuple] = None
        for (arm, ch), sp in sorted(stakes.items()):
            if self.pilots_per_arm.get(arm, 0) >= cap or (arm, "verify_fail") in self._verify_skip:
                continue
            try:
                post = self.model.posterior(arm, ch)
                m, sd = float(post.mean), float(post.sd)
            except Exception:
                continue
            if not (_orc_finite(m) and _orc_finite(sd)) or sd <= 0:
                continue
            zz = m / sd
            el = sp * (sd * math.exp(-0.5 * zz * zz) / math.sqrt(2.0 * math.pi) - m * norm_cdf(-zz))
            if best is None or el > best[0] + 1e-9:
                best = (el, arm, ch)
        if best is None:
            return None
        el, arm, ch = best
        cell_key = (arm[0], arm[1])
        subs = [sc for sc in self.dv.subs_of(cell_key)]
        avail = sorted(((int(sc.n) - int(self.used_subs.get(sc.key, 0)), sc.key) for sc in subs), reverse=True)
        if not avail or avail[0][0] < int(cfg.min_pilot):
            self._verify_skip.add((arm, "verify_fail"))
            return self._verify_spec(env) if len(self._verify_skip) < 50 else None
        n = int(min(int(cfg.max_pilot), avail[0][0], contacts))
        if n < int(cfg.min_pilot):
            return None
        if contacts - used >= n:
            v_contact = 0.0  # the plan leaves enough reach unused: pilot contacts displace nothing
        chans = list(getattr(self.dv, "channels", []) or [])
        p_ch = ch
        if float(self.dv.cost(ch)) * n > _ORC_VERIFY_MONEY_FRAC * el or float(self.dv.cost(ch)) * n > budget:
            p_ch = min(chans, key=lambda c: (float(self.dv.cost(c)), chans.index(c)))
        money = float(self.dv.cost(p_ch)) * n
        if money > budget or el <= money + n * v_contact:
            self.log.log("verify_skip", arm=arm_id(arm), channel=ch, expected_loss=round(el, 1),
                         pilot_cost=round(money + n * v_contact, 1))
            return None
        sub = avail[0][1]
        filters = {"filter_current_tariff": sub[0], "filter_arpu_segment": sub[1],
                   "filter_data_segment": sub[2], "filter_call_segment": sub[3]}
        return PilotSpec(arm=tuple(arm), channel=p_ch, n=n, sub=tuple(sub), filters=filters, score=float(el),
                         reason=f"verify: expected_loss={el:.1f} deploy_channel={ch} v_contact={v_contact:.1f}")

    @staticmethod
    def _env_counters(env: Any) -> tuple[float, int, int]:
        try:
            return float(env.remaining_budget), int(env.remaining_contacts), int(env.pilots_left)
        except Exception:
            return (float("nan"), 0, 0)

    def _account_env_delta(self, env: Any, pre: tuple[float, int, int], spec: Any) -> None:
        """If a failed pilot call still consumed resources, book them from env counter deltas."""
        post = self._env_counters(env)
        if post[2] >= pre[2]:
            return
        money = pre[0] - post[0] if _orc_finite(pre[0] - post[0]) else 0.0
        reach = max(pre[1] - post[1], 0)
        self.explore_money_spent += max(money, 0.0)
        self.explore_reach_spent += reach
        arm = tuple(spec.arm)
        self.pilots_per_arm[arm] = self.pilots_per_arm.get(arm, 0) + 1
        if spec.sub is not None and reach > 0:
            sub = tuple(spec.sub)
            self.used_subs[sub] = self.used_subs.get(sub, 0) + reach
        self.log.log("pilot_spent_without_result", arm=arm_id(arm), money=money, reach=reach)

    def _pilot_coverage(self) -> dict[tuple, tuple[float, float, Any]]:
        """sub -> (expected distinct pilot customers, pilot lift ratio, SubCell).

        Pilots draw random samples of the sub, so repeated pilots overlap: E[distinct] =
        n_sub·(1 − Π(1 − n_i/n_sub)). Ratio = max posterior ratio over the sub's pilots (best pilot lift kept).
        """
        out: dict[tuple, tuple[float, float, Any]] = {}
        subs = getattr(self.dv, "subs", {}) or {}
        fn = self._ratio_fn()
        for sub in sorted(set(self.used_subs) | set(self.pilot_records)):
            sc = subs.get(sub)
            if sc is None or int(sc.n) <= 0:
                continue
            n_sub = float(sc.n)
            recs = self.pilot_records.get(sub) or []
            if recs:
                miss = 1.0
                for _arm, _ch, n_i in recs:
                    miss *= max(1.0 - min(float(n_i), n_sub) / n_sub, 0.0)
                covered = n_sub * (1.0 - miss)
                r_p = max(fn(a[0], a[1], a[2], ch) for a, ch, _n in recs)
            else:  # contacts booked without arm info: assume disjoint, zero known pilot lift
                covered = min(float(self.used_subs.get(sub, 0)), n_sub)
                r_p = 0.0
            out[sub] = (covered, r_p, sc)
        return out

    def _pilot_overlap_gain(self, camps: list[dict]) -> float:
        """Gain the simulator over-counts on pilot customers re-contacted by final campaigns.

        Organizer scoring keeps max(pilot lift, final lift) per customer, while the final-only simulation counts
        the final lift f; the over-count per re-contacted customer is f − (max(p, f) − p) = min(p, f).
        """
        cov = self._pilot_coverage()
        if not cov:
            return 0.0
        fn = self._ratio_fn()
        total = 0.0
        for sub, (covered, r_p, sc) in cov.items():
            r_f: Optional[float] = None
            for camp in camps:
                if _orc_campaign_matches(camp, sub) and camp.get("target_tariff") != sub[0]:
                    r = fn(sub[0], sub[1], str(camp.get("target_tariff")), str(camp.get("channel")))
                    r_f = r if r_f is None else max(r_f, r)
            if r_f is None:
                continue
            total += covered * float(sc.mean_p) * min(r_p, r_f)
        return total

    def _allocate_sync(self, budget: float, contacts: int) -> tuple[Any, list[dict], dict]:
        """Try pilot-sub handling variants (overlap cost vs exclusion); keep the best simulated net."""
        cfg = self.cfg
        packer = self._c("CampaignPacker")(cfg, self.dv, self.sim)
        self.packer = packer
        variants: list[tuple[str, dict]] = []
        if self.used_subs:
            # Option cost already pays for re-contacting pilot customers; what is lost is their lift,
            # which the pilot already earned (organizer dedups by max). Charge that expected lost lift.
            cov = self._pilot_coverage()
            extra = {s: float(c) * float(sc.mean_p) * max(r_p, 0.0) for s, (c, r_p, sc) in sorted(cov.items())}
            variants.append(("overlap_cost", {"extra_cost": extra}))
            variants.append(("exclude", {"exclude_subs": set(self.used_subs)}))
        else:
            variants.append(("plain", {}))
        z_c = _ORC_COARSE_Z_FRAC * float(cfg.z_risk)
        if hasattr(packer, "pack_coarse"):
            variants.append(("coarse", {"_coarse": z_c}))
        best: tuple[tuple[int, float], str, Any, list[dict]] | None = None
        info: dict[str, Any] = {}
        for name, kw in variants:
            try:
                if "_coarse" in kw:
                    # cell-wide campaigns; the plan (evidence / veto) holds the options those campaigns deploy
                    camps = packer.pack_coarse(self.model, budget, contacts, z=kw["_coarse"])
                    plan = self._plan_from_campaigns(camps)
                else:
                    plan = self.allocator.allocate(budget, contacts, **kw)
                    camps = packer.pack(plan, self.model, budget, contacts)
                res = self._simulate(camps, budget, contacts)
                overlap = self._pilot_overlap_gain(camps)
                net = float(res.net) - overlap
                ok = bool(res.within_limits)
            except Exception as exc:
                self._fail(f"allocate_{name}", exc)
                continue
            info[name] = {"net": net, "overlap_gain": overlap, "campaigns": len(camps), "within_limits": ok}
            if not _orc_finite(net):
                continue
            rank = (1 if ok else 0, net)
            if best is None or rank[0] > best[0][0] or (rank[0] == best[0][0] and net > best[0][1] + 1e-9):
                best = (rank, name, plan, camps)
        if best is None:
            raise RuntimeError("all allocation variants failed")
        info["chosen"] = best[1]
        return best[2], best[3], info

    def _plan_from_campaigns(self, camps: list[dict]) -> Plan:
        """Plan whose options are the (sub, target, channel) triples the campaigns deploy (best per sub)."""
        best: dict[tuple, Any] = {}
        subs = getattr(self.dv, "subs", {}) or {}
        for camp in camps:
            tg, ch = str(camp.get("target_tariff")), str(camp.get("channel"))
            for sk in sorted(subs):
                if sk[0] == tg or not _orc_campaign_matches(camp, sk):
                    continue
                try:
                    opt = self.model.option(subs[sk], tg, ch)
                except Exception:
                    continue
                cur = best.get(sk)
                if cur is None or float(opt.net_mean) > float(cur.net_mean):
                    best[sk] = opt
        opts = [best[k] for k in sorted(best)]
        return Plan(options=opts, lambda_money=0.0, lambda_reach=0.0,
                    total_net_lcb=float(sum(o.net_lcb for o in opts)), total_cost=float(sum(o.cost for o in opts)),
                    total_contacts=int(sum(o.n for o in opts)))

    async def _stage_allocate(self, env: Any) -> None:
        if self.allocator is None or self.sim is None or self.model is None:
            self.stages["allocate"] = "skipped"
            return
        try:
            budget, contacts = float(env.remaining_budget), int(env.remaining_contacts)
            plan, camps, info = await asyncio.to_thread(self._allocate_sync, budget, contacts)
            self.plan, self.campaigns = plan, list(camps)
            self.extra["allocate"] = info
            self.log.log("allocate", campaigns=len(camps), options=len(getattr(plan, "options", []) or []), **{
                k: v for k, v in info.items() if k == "chosen"})
            self.stages["allocate"] = "ok"
        except Exception as exc:
            self._fail("allocate", exc)

    def _evidence(self, camps: list[dict], budget: float, contacts: int) -> list[dict]:
        """Per-campaign evidence: simulated economics + aggregated option uncertainty + pilots."""
        per: dict[str, dict] = {}
        try:
            res = self._simulate(camps, budget, contacts)
            per = {str(pc.get("name")): pc for pc in res.per_campaign}
        except Exception:
            per = {}
        options = list(getattr(self.plan, "options", []) or [])
        out: list[dict] = []
        for camp in camps:
            name = str(camp.get("campaign_name"))
            opts = [o for o in options if _orc_option_in_campaign(camp, o)]
            mean = sum(float(o.net_mean) for o in opts)
            # options sharing (tariff, arpu, target, channel) share one posterior -> perfectly correlated
            grp: dict[tuple, float] = {}
            for o in opts:
                g = (o.sub[0], o.sub[1], o.target, o.channel)
                grp[g] = grp.get(g, 0.0) + abs(float(o.net_sd))
            sd = math.sqrt(sum(v * v for v in grp.values()))
            p_pos = norm_cdf(mean / sd) if sd > 0 else (1.0 if mean > 0 else 0.0)
            arms = sorted({(o.sub[0], o.sub[1], o.target) for o in opts})
            n_pilots = sum(self.pilots_per_arm.get(a, 0) for a in arms)
            posts = []
            for a in arms[:6]:
                try:
                    p = self.model.posterior(a, camp.get("channel"))
                    posts.append({"arm_id": arm_id(a), "mean": float(p.mean), "sd": float(p.sd),
                                  "n_obs": int(self.model.n_obs(a))})
                except Exception:
                    continue
            pc = per.get(name, {})
            out.append({
                "name": name, "campaign_id": name, "target_tariff": camp.get("target_tariff"), "channel": camp.get("channel"),
                "filters": {k: camp.get(k) for k in _ORC_FILTER_KEYS},
                "n_contacted": pc.get("n_contacted"), "cost": pc.get("cost"), "gross": pc.get("gross"),
                "net_mean": mean, "net_sd": sd, "p_pos": p_pos, "n_pilots": n_pilots, "posteriors": posts,
            })
        return out

    async def _stage_review(self, env: Any, mode: str, hard_deadline: float) -> None:
        """LLM risk review with what-if simulation; veto applied (and re-packed) only in decide mode."""
        if not self.campaigns or self.llm is None or self.sim is None:
            self.stages["review"] = "skipped"
            return
        budget, contacts = float(env.remaining_budget), int(env.remaining_contacts)
        camps = list(self.campaigns)
        try:
            evidence = await asyncio.to_thread(self._evidence, camps, budget, contacts)

            def what_if(veto_names: list[str]) -> dict:
                veto = {str(v) for v in (veto_names or [])}
                kept = [c for c in camps if str(c.get("campaign_name")) not in veto]
                r = self._simulate(kept, budget, contacts)
                return {"net": float(r.net) - self._pilot_overlap_gain(kept), "gross": float(r.gross), "cost": float(r.cost), "contacts": int(r.contacts)}

            timeout = min(float(self.cfg.llm_call_timeout_s) * 2.0, max(hard_deadline - time.monotonic() - 5.0, 0.0))
            if timeout <= 0:
                self.stages["review"] = "skipped"
                return
            outcome = await asyncio.wait_for(self.llm.review_plan(camps, evidence, what_if), timeout=timeout)
            self.review = outcome
            names = {str(c.get("campaign_name")) for c in camps}
            veto = sorted({str(v) for v in (outcome.veto or []) if str(v) in names})
            self.log.log("review", source=outcome.source, applied=bool(outcome.applied), veto=veto)
            self.extra["review"] = {"source": outcome.source, "applied": bool(outcome.applied), "veto": veto,
                                    "summary": str(outcome.summary)[:1000], "rationale": str(outcome.rationale)[:1000]}
            if mode == "decide" and outcome.applied and veto and len(veto) < len(camps):
                self.campaigns = await asyncio.to_thread(self._apply_veto, camps, veto, budget, contacts)
                self.log.log("veto_applied", veto=veto, campaigns=len(self.campaigns))
            self.stages["review"] = "ok"
        except Exception as exc:
            self._fail("review", exc)

    def _apply_veto(self, camps: list[dict], veto: list[str], budget: float, contacts: int) -> list[dict]:
        """Drop vetoed campaigns' options and re-pack; fall back to plain filtering."""
        vset = set(veto)
        kept = [c for c in camps if str(c.get("campaign_name")) not in vset]
        vetoed = [c for c in camps if str(c.get("campaign_name")) in vset]
        if (self.extra.get("allocate") or {}).get("chosen") == "coarse":
            return kept  # cell-wide campaigns are independent: dropping them is the re-pack
        try:
            opts = [o for o in self.plan.options if not any(_orc_option_in_campaign(c, o) for c in vetoed)]
            plan2 = Plan(options=opts, lambda_money=self.plan.lambda_money, lambda_reach=self.plan.lambda_reach,
                         total_net_lcb=float(sum(o.net_lcb for o in opts)),
                         total_cost=float(sum(o.cost for o in opts)),
                         total_contacts=int(sum(o.n for o in opts)))
            repacked = self.packer.pack(plan2, self.model, budget, contacts)
            if repacked:
                self.plan = plan2
                return list(repacked)
        except Exception as exc:
            self._fail("repack", exc)
        return kept

    def _guardrails(self, sim: Any) -> Any:
        """Guardrails instance (passes the run log when the implementation accepts it)."""
        cls = self._c("Guardrails")
        try:
            return cls(self.cfg, self.dv, sim, log=self.log)
        except TypeError:
            return cls(self.cfg, self.dv, sim)

    def _finalize(self, env: Any, campaigns: list[dict]) -> list[dict]:
        """Guardrails validation (fallback: emergency campaign) + best-effort report."""
        final: list[dict] = []
        try:
            budget, contacts = float(env.remaining_budget), int(env.remaining_contacts)
            if self.dv is None:
                raise RuntimeError("no dataview")
            sim = self.sim if self.sim is not None else self._c("ScoreSimulator")(self.dv, env.customer_profile)
            final = list(self._guardrails(sim).validate(list(campaigns), budget, contacts))
            self.stages["finalize"] = "ok"
        except Exception as exc:
            self._fail("finalize", exc)
            final = _orc_emergency_campaigns(env)
        try:
            if self.sim is not None and self.model is not None and final:
                r = self._simulate(final, float(env.remaining_budget), int(env.remaining_contacts))
                self.extra["final_sim"] = {"net": float(r.net), "gross": float(r.gross), "cost": float(r.cost),
                                           "contacts": int(r.contacts), "within_limits": bool(r.within_limits)}
        except Exception:
            pass
        self.extra["stages"] = dict(self.stages)
        self.extra["explore"] = {"pilots": len([p for p in self.log.pilots if "error" not in p]),
                                 "money": self.explore_money_spent, "reach": self.explore_reach_spent}
        self.log.log("final", campaigns=len(final))
        try:
            path = self.cfg.report_path
            if path and not os.path.isabs(path):
                path = os.path.join(_orc_base_dir(), path)
            if path:
                self._c("Reporter")(self.cfg, path).write(self.log, final, dict(self.extra))
        except Exception as exc:
            self._fail("report", exc)
        return final


class Agent:
    """Submission entry point: Agent().act(env) -> list of campaign dicts (never raises)."""

    def __init__(
        self,
        cfg: Optional[Config] = None,
        components: Optional[dict[str, Any]] = None,
        llm_mode: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.components = components
        self.llm_mode = llm_mode
        self.last_orchestrator: Optional[Orchestrator] = None

    def act(self, env: Any) -> list[dict]:
        """Run the full pipeline; on any failure return the guardrails fallback list."""
        try:
            _orc_load_dotenv()
        except Exception:
            pass
        cfg = self.cfg
        try:
            if cfg is None:
                cfg = Config.from_env()
        except Exception:
            cfg = Config()
        orch = Orchestrator(cfg, components=self.components, llm_mode=self.llm_mode)
        self.last_orchestrator = orch
        coro = None
        try:
            coro = orch.run(env)
            out = run_coro(coro)
            if isinstance(out, list):
                return out
            raise TypeError("pipeline returned non-list")
        except Exception as exc:
            if coro is not None:
                try:
                    coro.close()
                except Exception:
                    pass
            try:
                orch.log.log("act_error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
            except Exception:
                pass
            return self._fallback(cfg, orch, env)

    def _fallback(self, cfg: Config, orch: Orchestrator, env: Any) -> list[dict]:
        """Guardrails fallback (empty plan validated) or an emergency no-op campaign."""
        try:
            dv = orch.dv if orch.dv is not None else orch._c("DataView")(env.customer_profile, env.tariffs, env.channels)
            sim = orch.sim if orch.sim is not None else orch._c("ScoreSimulator")(dv, env.customer_profile)
            out = orch._c("Guardrails")(cfg, dv, sim).validate([], float(env.remaining_budget),
                                                                  int(env.remaining_contacts))
            if isinstance(out, list):
                return out
        except Exception:
            pass
        return _orc_emergency_campaigns(env)
