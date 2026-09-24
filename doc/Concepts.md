# Kairos Temporal Flow --- Concepts

---

## 1. The Big Picture

The goal is to attach temporal behaviour to each spatial voxel cell in the scene graph: "at this cell, pedestrians tend to walk east in the morning and north in the evening, and there are more of them on weekdays".

The system does this in two layers that operate independently:

```
Incoming detections (theta, rho, xyz, t)
        │
        ▼
┌──────────────┐    every detection
│   GMM layer  │ ← models the SHAPE of the (theta, rho) distribution:
│   (mixture)  │   which directions/speeds currently exist at this cell
└──────┬───────┘
       │  responsibility r[k] per component (softmax-style weight)
       ▼
┌──────────────┐    one sample per crossing (weights)
│  NUDFT layer │ ← models HOW EACH COMPONENT'S PARAMETERS CHANGE OVER TIME:
│  (per param) │   is component k more or less active at this hour of day?
└──────────────┘
```

The GMM answers "what is the distribution right now?".
The NUDFT answers "how does that distribution modulate over time?".

---

## 2. The GMM Layer

Eight direction slots are pre-assigned at k * π/4 (east, NE, north, ...). Their headings and covariances are hardcoded.

You can visualise each component as a tent (Gaussian) in `(theta, rho)` space. The tents are narrow and fixed in place: their headings and covariances stay constant for the life of the map.

The only free parameter is the **mixing weight π_k** for each slot: how much of the current flow belongs to direction k. This is what the NUDFT learns over time.

Each slot also keeps its own mean speed μ_ρ: the running mean of the speeds routed to it. Only the mean term of its predictor is read, since speed varies little with time of day against the width of the slot's speed spread.

---

## 3. The NUDFT Layer

### What NUDFT tracks

Per-slot k:

| NUDFT instance | Tracks |
|---|---|
| pi[k] | Mixing weight: is this component active right now? |
| mu_rho[k] | Running mean speed of the slot (its mean term only is read) |

Headings stay pinned at k·π/4 and covariances at `fixed_sigma_*`; the NUDFTs track only the two channels above.

### What gamma0 and gammas mean

Each NUDFT is an online running Fourier estimator. For a tracked parameter `y`:

```
gamma0    = time-averaged value of y  (DC / mean term)
gammas[f] = complex coefficient at omega[f] = 2*pi / period[f]
           = encodes amplitude and phase of oscillation at that period
```

Prediction at time t:

```
predict(t) = gamma0 + sum_{top order} |gammas[f]| * cos(omega[f]*t + phase[f])
```

`gamma0` is the always-available baseline answer. `gammas` are temporal corrections. The NUDFT update is a running average, each new observation
has weight `1/(n+1)`, so every observation counts equally and the estimate settles as n grows.

---

## 4. Candidate Periods and Dataset Length

Each NUDFT frequency is `omega[f] = 2*pi / period[f]`. For a Fourier fit to be reliable you need at least 2–3 full cycles in the training data.