#set page(
  paper: "us-letter",
  margin: (x: 1.8cm, top: 2.0cm, bottom: 2.0cm),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8pt, fill: luma(100), font: "New Computer Modern")
      #align(center)[_Comparative Analysis of Nature-Inspired Cache Replacement in Cloud Computing_]
      #v(-0.3em)
      #line(length: 100%, stroke: 0.4pt + luma(180))
    ]
  },
  footer: context {
    set text(size: 9pt, font: "New Computer Modern")
    align(center)[#counter(page).get().first()]
  }
)

#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.54em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => block(
  above: 1.1em,
  below: 0.6em,
  sticky: true,
  [
    #set text(size: 10.5pt, weight: "bold")
    #smallcaps(it)
  ]
)

#show heading.where(level: 2): it => block(
  above: 0.9em,
  below: 0.5em,
  sticky: true,
  [
    #set text(size: 9.5pt, weight: "bold", style: "italic")
    #it
  ]
)

// Title & Meta
#align(center)[
  #v(0.2em)
  #text(size: 16pt, weight: "bold")[Comparative Analysis of Nature-Inspired Optimization Algorithms for Cache Replacement in Cloud Computing]
  
  #v(0.8em)
  #text(size: 11pt, weight: "bold")[Shagun Agarwal] \
  #text(size: 9pt, style: "italic")[Department of Computer Science and Engineering] \
  #text(size: 8.5pt)[author\@institution.edu]
  #v(0.8em)
]

// Abstract and Keywords
#align(center)[
  #block(width: 95%, stroke: (top: 0.6pt + luma(150), bottom: 0.6pt + luma(150)), inset: (y: 0.8em))[
    #set par(justify: true, leading: 0.48em)
    #set text(size: 8.5pt)
    *Abstract*---Efficient cache replacement in cloud computing systems is paramount for minimizing data retrieval latencies, avoiding Service Level Agreement (SLA) violation penalties, and mitigating I/O bottlenecks across multi-tier storage architectures (DRAM and SSDs). Traditional heuristic caching policies such as Least Recently Used (LRU), Least Frequently Used (LFU), and Random Replacement (RR) rely on rigid, myopic heuristics that fail to adapt to shifting request distributions and differentiated Quality-of-Service (QoS) priorities. In this paper, we present an experimental comparative study of nature-inspired metaheuristic optimization algorithms engineered for predictive multi-tier cache replacement. We formulate cache replacement as a continuous multi-factor predictive utility optimization problem and evaluate ten algorithms across three tiers: (i) newer nature-inspired algorithms (SFO, DSOA-EC, HO); (ii) established metaheuristics (GA, PSO, ABC, WOA); and (iii) traditional deterministic baselines (LRU, LFU, RR). Using a strict three-way split evaluation protocol (10k training, 40k validation, 40k holdout) on real cloud traces from the SNIA repository alongside synthetic stationary Zipf, bursty, trending, and concept-drift workloads, we benchmark all algorithms under identical multi-tier hardware constraints and QoS SLA penalty profiles. Our findings demonstrate that metaheuristic-tuned policies decisively outperform traditional baselines, improving overall Cache Hit Ratio (CHR) by up to 22.17 percentage points on Zipfian distributions and reducing SLA penalty costs by more than 50%. Among optimizers, PSO and HO achieved the highest holdout CHR (67.62%) on real cloud traces, while SFO and DSOA-EC delivered superior High-Priority Content (HPC) retention (up to 70.11% CHR). Furthermore, we evaluate cross-workload transferability, parameter reproducibility across random seeds, and runtime efficiency to guide production deployments.

    #v(0.4em)
    *Keywords*---Cloud Computing, Multi-Tier Caching, Nature-Inspired Metaheuristics, Cache Replacement, Quality of Service (QoS), Swarm Intelligence.
  ]
]

#v(0.6em)

// Two-column body
#show: rest => columns(2, gutter: 1.6em, rest)

= Introduction

Cloud computing infrastructures host diverse, large-scale workloads ranging from enterprise databases to latency-sensitive microservices. In distributed cloud data centers, high-volume data retrieval constitutes a primary operational bottleneck due to the widening throughput gap between processing units and secondary storage, commonly termed the "I/O wall" [1].

To bridge this latency gap, cloud architectures deploy multi-tiered memory hierarchies combining high-speed, scarce Dynamic Random-Access Memory (DRAM) with high-capacity Solid-State Drives (SSDs) [1, 2]. Because aggregate working sets exceed physical memory capacity, cache replacement algorithms must continuously select which resident items to retain and which to evict.

Cache replacement performance directly governs:
- *Cache Hit Ratio (CHR):* Fraction of requests served directly from high-speed cache.
- *End-to-End Retrieval Latency:* Response time dictated by nanosecond-scale DRAM, microsecond-scale SSD, or millisecond-scale remote fetches.
- *Quality-of-Service (QoS) and SLA Penalties:* Cloud service level agreements impose heavy contractual penalties for delayed high-priority requests [1].

Traditional deterministic baselines such as LRU, LFU, and Random Replacement use one-dimensional heuristics (recency or cumulative frequency alone) and are oblivious to client QoS priorities and multi-tier storage asymmetry [1, 2].

To address these limitations, we evaluate parameter-tunable predictive utility caching, where candidate policies optimize continuous weights governing recency, frequency, temporal aging decay, QoS priorities, and SLA penalty costs. We conduct a rigorous comparative study of ten algorithms across three categories: newer nature-inspired algorithms (SFO, DSOA-EC, HO), established metaheuristics (GA, PSO, ABC, WOA), and traditional baselines (LRU, LFU, RR), evaluated on real and synthetic cloud traces under zero-leakage validation across six focused research questions.

= Problem Statement & System Model

Consider a cloud Content Store comprising a two-tier memory hierarchy:
$ cal(C) = cal(C)_"DRAM" union cal(C)_"SSD" $
with total capacity $B_"total"$ bytes, partitioned such that DRAM occupies $B_"DRAM" = 0.20 dot B_"total"$ and SSD occupies $B_"SSD" = 0.80 dot B_"total"$. Hardware parameters follow realistic cloud storage profiles: DRAM exhibits access latency $tau_"DRAM" = 100" ns"$ with $40" GB/s"$ read and $20" GB/s"$ write bandwidth; SSD exhibits $tau_"SSD" = 10 mu"s"$ with $2" GB/s"$ read and $1" GB/s"$ write bandwidth [1].

A request stream $cal(R) = {r_1, r_2, dots, r_N}$ arrives over time, where each request $r_k$ has an object key $o_k$, size $s_k$, priority class $p_k in {"HPC", "LPC"}$ (High-Priority vs. Low-Priority Content), and timestamp $t_k$.

== QoS SLA Miss Penalty Model
When an item is retrieved, the response time $T_"resp"$ is computed based on tier transfer times. The cloud provider incurs a contractual SLA penalty $P(r_k)$ matching the multi-tier QoS cloud benchmark model [1]:
$ P(r_k) = cases(
  5  &"if " p_k = "HPC" and T_"resp" = 0,
  0  &"if " T_"resp" < 0.02" s",
  50 &"if " p_k = "HPC" and 0.02" s" <= T_"resp" < 0.15" s",
  75 &"if " p_k = "HPC" and T_"resp" >= 0.15" s",
  10 &"if " p_k = "LPC" and 0.02" s" <= T_"resp" < 0.15" s",
  15 &"if " p_k = "LPC" and T_"resp" >= 0.15" s",
) $
HPC requests induce up to five times greater penalties than LPC requests ($75$ vs. $15$).

== Multi-Objective Fitness Function
During hyperparameter optimization, candidate configurations are evaluated on a validation split using the balanced objective:
$ cal(F) = 0.40 dot "CHR"_"total" + 0.35 dot "CHR"_"HPC" - 0.15 dot tilde(P) - 0.10 dot tilde(L) $
where $tilde(P) = (sum P(r_k)) / (75 dot N)$ and $tilde(L) = min(1.0, bar(T)_"resp" / 0.100" s")$ normalize penalty and latency against baseline bounds.

= Literature Review

Early foundational caching research focused on theoretical bounds, such as Belady's optimal MIN algorithm [3]. Practical online algorithms relied on recency (LRU) or frequency (LFU). Megiddo and Modha proposed the Adaptive Replacement Cache (ARC) [2], which dynamically tunes a balance parameter between recency and frequency lists.

In cloud computing, Ait-Oucheggou et al. (2025) proposed QM-ARC [1], establishing the two-tier QoS-aware Content Store framework for Named Data Networking and cloud storage. Their study showed that heuristic and reinforcement learning policies mitigate SLA penalties. However, metaheuristics for continuous utility parameter tuning across multi-tier storage architectures remain underexplored.

Swarm and evolutionary optimization algorithms have demonstrated strong capabilities across non-convex search spaces. Holland established Genetic Algorithms (GA) [4]. Kennedy and Eberhart introduced Particle Swarm Optimization (PSO) [5]. Karaboga and Basturk developed the Artificial Bee Colony (ABC) [6], and Mirjalili and Lewis introduced the Whale Optimization Algorithm (WOA) [7]. Recently, newer biological metaphors have been proposed, including Dove Swarm Optimization (DSOA) by Su et al. [8], Hippopotamus Optimization (HO) by Amiri et al. [9], and Sharpbelly Fish Optimization (SFO). This paper evaluates these metaheuristics against classical policies in a matched multi-tier cloud testbed.

#v(0.3em)
#align(center)[
  #block(width: 100%)[
    #set text(size: 8pt)
    *Table 1:* Evaluated Cache Replacement Algorithms.
    #v(0.2em)
    #table(
      columns: (1.1fr, 0.9fr, 2fr),
      align: (left, left, left),
      stroke: (x, y) => if y == 0 { (bottom: 0.6pt + black, top: 0.6pt + black) } else if y == 10 { (bottom: 0.6pt + black) } else { none },
      table.header([*Category*], [*Algorithm*], [*Search Mechanism*]),
      [Newer Bio-Inspired], [SFO], [School aggregation, fast predatory rush, dispersal],
      [Newer Bio-Inspired], [DSOA-EC], [Satiety decay, exploration-competition],
      [Newer Bio-Inspired], [HO], [Herd defense, Lévy-flight evasions],
      [Established], [GA], [Tournament selection, blend crossover, mutation],
      [Established], [PSO], [Velocity vectors, $p_"best"$ and $g_"best"$ tracking],
      [Established], [ABC], [Employed, onlooker, and scout bee foraging],
      [Established], [WOA], [Bubble-net spiral and encircling prey maneuvers],
      [Classical Baseline], [LRU], [Recency ordering with tier demotion],
      [Classical Baseline], [LFU], [Reference count tracking across tiers],
      [Classical Baseline], [RR], [Uniform random victim selection],
    )
  ]
]

= Algorithms & Predictive Utility Model

== Predictive Utility Formula in the Implementation
In our caching policy, each candidate policy configuration is represented as a 10-dimensional parameter vector:
$ vec(theta) = [w_"rec", w_"freq", w_"decay", w_"size", w_"prio", w_"pen", w_"reuse", theta_"DRAM", alpha_"QoS", p_"target"] $

When eviction or admission is evaluated for resident item $o_i$ at current timestamp $t$ (with access count $C_i$, last access $t_i^"last"$, inter-arrival average $bar(Delta)_i$, and priority $p_i$):
1. *Recency with Temporal Decay:*
   $ f_"rec" = 1 / (1 + max(0.001, w_"decay") dot (t - t_i^"last")) $
2. *Logarithmic Frequency:*
   $ f_"freq" = ln(1 + C_i) $
3. *QoS Priority Feature:*
   $ f_"prio" = cases(1.0 &"if " p_i = "HPC", 0.0 &"if " p_i = "LPC") $
4. *SLA Miss Penalty Weight:*
   $ f_"pen" = cases(1.0 &"if " p_i = "HPC", 0.20 &"if " p_i = "LPC") $
5. *Normalized Object Size:*
   $ f_"size" = s_i / max(s_"slot", 1) $
6. *Predictive Inter-Arrival Reuse Likelihood:*
   $ f_"reuse" = exp(- (t - t_i^"last") / max(bar(Delta)_i, 0.005)) $

The core utility and modulated predictive score are computed as:
$ U_"core" = w_"rec" + w_"freq" f_"freq" + w_"prio" f_"prio" + w_"pen" f_"pen" + w_"reuse" f_"reuse" $
$ U(o_i, t) = U_"core" dot f_"rec" + w_"size" f_"size" $

== DRAM Admission and Eviction Logic
When an object is admitted or requested:
$ "Placement Tier" = cases(
  "DRAM" &"if " U(o_i, t) >= theta_"DRAM",
  "SSD"  &"otherwise"
) $
To ensure $O(1)$ constant-time operational overhead during online execution, eviction selects the minimum utility item from a random tournament sample of $K = 24$ resident blocks.

= Proposed Methodology

The experimental framework enforces a strict zero-leakage evaluation protocol:
1. *Trace Dataset:* SNIA cloud block storage trace `cluster07` (90,000 requests) and synthetic workloads (Zipf $alpha = 1.2$, $5times$ bursty, and $alpha = 0.8 -> 1.6$ drift).
2. *QoS Preprocessing:* Deterministic 30% HPC / 70% LPC tagging ($"seed" = 42$).
3. *Request Splitting:* First 10,000 requests for training; next 40,000 requests for validation tuning; final 40,000 requests for untouched holdout evaluation.
4. *Cache Sizing:* Total capacity fixed to exactly 10% of the training unique catalog (20% DRAM, 80% SSD).
5. *Search Budget:* 30 agents across 25 iterations for each of the 7 metaheuristics, selecting the configuration with the highest validation fitness.
6. *Holdout Benchmark:* The selected configuration is frozen and evaluated once on the untouched holdout partition alongside LRU, LFU, and RR baselines.

= Experimental Results

== Real Cloud Trace Matched Holdout Benchmark (RQ1)
Table 2 reports holdout results on the untouched 40,000 requests (requests 50,001--90,000) of the SNIA `cluster07` trace.

#v(0.3em)
#align(center)[
  #block(width: 100%)[
    #set text(size: 8pt)
    *Table 2:* Matched Holdout Results on Real Cloud Trace (`cluster07`).
    #v(0.2em)
    #table(
      columns: (1.1fr, 1fr, 1fr, 1fr, 1.1fr, 1.1fr),
      align: (left, center, center, center, center, center),
      stroke: (x, y) => if y == 0 { (bottom: 0.6pt + black, top: 0.6pt + black) } else if y == 7 { (bottom: 0.6pt + black) } else { none },
      table.header([*Policy*], [*Total CHR*], [*HPC CHR*], [*LPC CHR*], [*Penalty*], [*Latency*]),
      [*PSO*], [*67.62%*], [69.96%], [*66.59%*], [*309,275*], [*34.37 ms*],
      [*SFO*], [66.99%], [*70.11%*], [65.61%], [311,155], [34.90 ms],
      [*GA*], [67.38%], [69.96%], [66.24%], [310,920], [34.61 ms],
      [*DSOA*], [67.13%], [70.05%], [65.84%], [310,710], [34.93 ms],
      [*ABC*], [67.26%], [69.69%], [66.19%], [310,965], [34.74 ms],
      [*HO*], [*67.62%*], [69.61%], [66.75%], [312,470], [34.41 ms],
      [*WOA*], [67.41%], [69.76%], [66.38%], [311,530], [34.52 ms],
    )
  ]
]

- *Overall Hit Ratio:* PSO and HO achieved the highest holdout CHR of *67.62%*. PSO yielded the lowest overall SLA penalty (309,275) and lowest average retrieval time (34.37 ms).
- *QoS Protection:* SFO achieved the highest HPC hit ratio (*70.11%*), closely followed by DSOA (*70.05%*).

== Synthetic Workload Robustness (RQ2)
To examine adaptation under controlled request dynamics, we evaluate all ten algorithms across synthetic workload families:
1. *Static Heavy-Tailed Skew (Zipf $alpha = 1.2$):* High temporal locality with stationary popularities. GA and PSO achieved CHRs of 59.50% and 59.33%, retaining frequently requested catalog heads.
2. *Bursty Popularity Spikes ($5times$ base rate):* Simulates transient flash crowds. HO achieved the peak CHR of *62.50%* with an average latency of 34.39 ms, followed closely by PSO (62.33%) and GA (61.83%).
3. *Exponential Trending ($8times$ drift):* Tests rapid item life-cycle transitions. DSOA-EC achieved the top validation score (0.2838), where its satiety decay mechanism proactively demoted items undergoing popularity decline.

== Competitiveness Against Classical Baselines (RQ7)
Table 3 compares the optimized policies against LRU, LFU, and Random across three synthetic workload families.

#v(0.3em)
#align(center)[
  #block(width: 100%)[
    #set text(size: 8pt)
    *Table 3:* Baseline Competitiveness Across Workload Families.
    #v(0.2em)
    #table(
      columns: (1.3fr, 1.1fr, 1.1fr, 1.1fr, 1.1fr, 1fr),
      align: (left, left, center, center, center, center),
      stroke: (x, y) => if y == 0 { (bottom: 0.6pt + black, top: 0.6pt + black) } else if y in (10, 20, 30) { (bottom: 0.4pt + luma(180)) } else { none },
      table.header([*Workload*], [*Policy*], [*CHR*], [*Penalty*], [*Latency*], [*Score*]),
      [Static Zipf], [GA], [59.50%], [5,295], [35.54 ms], [*0.4026*],
      [Static Zipf], [PSO], [59.33%], [6,195], [36.16 ms], [0.3445],
      [Static Zipf], [SFO], [58.83%], [6,255], [36.62 ms], [0.3417],
      [Static Zipf], [DSOA], [54.17%], [6,425], [38.78 ms], [0.3248],
      [Static Zipf], [ABC], [56.33%], [5,490], [37.70 ms], [0.3913],
      [Static Zipf], [HO], [58.50%], [6,220], [36.60 ms], [0.3428],
      [Static Zipf], [WOA], [58.50%], [6,155], [36.64 ms], [0.3474],
      [Static Zipf], [LRU], [37.33%], [10,670], [50.93 ms], [0.0255],
      [Static Zipf], [LFU], [46.83%], [8,855], [44.34 ms], [0.1612],
      [Static Zipf], [Random], [11.00%], [10,780], [60.07 ms], [-0.0658],
      [Bursty (5x)], [GA], [61.83%], [5,550], [34.71 ms], [0.3864],
      [Bursty (5x)], [PSO], [62.33%], [5,475], [34.53 ms], [0.3913],
      [Bursty (5x)], [SFO], [60.17%], [5,450], [35.90 ms], [0.3908],
      [Bursty (5x)], [DSOA], [58.50%], [5,520], [37.57 ms], [0.3852],
      [Bursty (5x)], [ABC], [60.83%], [5,340], [35.39 ms], [*0.3988*],
      [Bursty (5x)], [HO], [*62.50%*], [5,480], [*34.39 ms*], [0.3801],
      [Bursty (5x)], [WOA], [58.33%], [5,395], [37.70 ms], [0.3920],
      [Bursty (5x)], [LFU], [53.67%], [6,730], [40.97 ms], [0.2878],
      [Bursty (5x)], [LRU], [45.50%], [8,365], [44.97 ms], [0.1777],
      [Bursty (5x)], [Random], [11.33%], [10,185], [58.72 ms], [-0.0388],
      [Drift (0.8-1.6)], [DSOA], [*91.44%*], [3,380], [*8.44 ms*], [*0.6636*],
      [Drift (0.8-1.6)], [PSO], [91.31%], [3,415], [8.72 ms], [0.6630],
      [Drift (0.8-1.6)], [ABC], [90.94%], [3,440], [9.01 ms], [0.6623],
      [Drift (0.8-1.6)], [SFO], [91.25%], [3,455], [8.70 ms], [0.6616],
      [Drift (0.8-1.6)], [WOA], [90.56%], [*3,360*], [8.60 ms], [0.6611],
      [Drift (0.8-1.6)], [GA], [90.62%], [3,480], [9.46 ms], [0.6608],
      [Drift (0.8-1.6)], [HO], [90.12%], [3,530], [9.21 ms], [0.6586],
      [Drift (0.8-1.6)], [LFU], [90.56%], [3,780], [9.26 ms], [0.6501],
      [Drift (0.8-1.6)], [LRU], [89.69%], [4,795], [10.28 ms], [0.6346],
      [Drift (0.8-1.6)], [Random], [75.25%], [8,130], [22.29 ms], [0.5512],
    )
  ]
]

#v(0.3em)
#align(center)[
  #image("paper_results_bundle/rq7_baseline_competitiveness/results/figures/rq7_static_zipf_alpha_12_comparison.png", width: 95%)
  #text(size: 7.5pt, style: "italic")[*Figure 1:* Baseline competitiveness comparison on Static Zipf ($alpha = 1.2$).]
]

On the stationary Zipf workload, GA achieves *59.50%* CHR versus *37.33%* for LRU (+22.17% gain) and cuts SLA penalties by half (5,295 vs. 10,670). Under $5times$ bursts, HO achieves peak CHR (*62.50%*) versus 45.50% for LRU.

== Multi-Workload Drift, Efficiency, and Stability (RQ3, RQ4, RQ5)
Table 4 presents a synthesis of algorithm behavior across distribution drift (RQ3), optimization runtime (RQ4), and random-seed reproducibility over 5 seeds (RQ5).

#v(0.3em)
#align(center)[
  #block(width: 100%)[
    #set text(size: 8pt)
    *Table 4:* Convergence Runtime, Drift Score, and Seed Stability.
    #v(0.2em)
    #table(
      columns: (1.1fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
      align: (left, center, center, center, center),
      stroke: (x, y) => if y == 0 { (bottom: 0.6pt + black, top: 0.6pt + black) } else if y == 7 { (bottom: 0.6pt + black) } else { none },
      table.header([*Policy*], [*Mean Runtime*], [*Drift Score*], [*Seed Fitness*], [*Fitness Std*]),
      [*GA*], [*11.67 s*], [0.5216], [0.4570], [*0.0004*],
      [*PSO*], [11.85 s], [0.5255], [0.4576], [0.0006],
      [*SFO*], [12.16 s], [0.5244], [*0.4593*], [0.0011],
      [*WOA*], [13.99 s], [0.5214], [0.4514], [0.0134],
      [*HO*], [17.01 s], [0.5223], [0.4550], [0.0017],
      [*DSOA*], [17.98 s], [0.5229], [0.4549], [0.0050],
      [*ABC*], [41.90 s], [*0.5256*], [0.4585], [0.0051],
    )
  ]
]

- *Efficiency:* GA, PSO, and SFO converged fastest ($approx 11$--$12$ s), whereas ABC required $41.90$ s on average due to evaluating three bee phases per cycle.
- *Drift Robustness:* ABC and PSO achieved the highest scores on distribution shift ($0.5256$ and $0.5255$).
- *Reproducibility:* GA and PSO exhibited the lowest variance across random seeds ($sigma = 0.0004$ and $0.0006$), while SFO achieved the highest mean fitness ($0.4593$).

== Cross-Workload Generalization and Transfer (RQ6)
In cloud production environments, caching policies trained on one service trace are frequently deployed across different microservices or fluctuating traffic regimes without retraining. Table 5 evaluates cross-workload generalization across all source-target pairs, measuring transfer CHR, transfer fitness $cal(F)_"trans"$, same-domain fitness $cal(F)_"same"$, and the generalization gap $Delta = cal(F)_"trans" - cal(F)_"same"$.

#v(0.3em)
#align(center)[
  #block(width: 100%)[
    #set text(size: 8pt)
    *Table 5:* RQ6 Cross-Workload Generalization Matrix across 6 Transfer Pairs.
    #v(0.2em)
    #table(
      columns: (1.5fr, 0.8fr, 0.9fr, 1fr, 1fr, 0.9fr),
      align: (left, left, center, center, center, center),
      stroke: (x, y) => if y == 0 { (bottom: 0.6pt + black, top: 0.6pt + black) } else if y == 6 { (bottom: 0.6pt + black) } else { none },
      table.header([*Source -> Target*], [*Top Policy*], [*Trans. CHR*], [$cal(F)_"trans"$], [$cal(F)_"same"$], [*Gap $Delta$*]),
      [Static Zipf -> Bursty], [DSOA], [47.8%], [0.3115], [0.3118], [+0.0004],
      [Static Zipf -> Drift], [WOA], [79.6%], [0.5849], [0.6519], [+0.0670],
      [Bursty -> Static Zipf], [PSO], [55.2%], [0.3967], [0.3924], [-0.0043],
      [Bursty -> Drift], [GA], [79.8%], [0.5858], [0.6547], [+0.0689],
      [Drift -> Static Zipf], [SFO], [64.0%], [0.4525], [0.3937], [-0.0589],
      [Drift -> Bursty], [WOA], [60.5%], [0.3879], [0.3146], [-0.0733],
    )
  ]
]

Key generalization findings:
- *Positive Transfer from Dynamic to Static:* Policies trained on complex dynamic workloads (such as Drift or Bursty Trending) transferred exceptionally well to Static Zipf. SFO achieved a transfer hit ratio of *64.0%* on Static Zipf (exceeding even native in-domain training by 5 percentage points), demonstrating that training on drift workloads discovers robust, generalizable utility weights.
- *Minimal Degradation under Domain Shifts:* When transferring between bursty and stationary workloads, the performance gap $Delta$ remained within $0.004$, confirming that the 10-dimensional utility parameterization captures fundamental locality traits rather than overfitting to specific request keys.

#v(0.3em)
#align(center)[
  #image("paper_results_bundle/rq4_efficiency/static_zipf_rq4_runtime_quality.png", width: 95%)
  #text(size: 7.5pt, style: "italic")[*Figure 2:* Runtime vs. solution quality trade-off across algorithms on Static Zipf.]
]

= Discussion

== Why Metaheuristics Outperform Traditional Baselines
1. *Multi-Attribute Utility Integration:* Classical LRU and LFU rely on isolated metrics that are easily corrupted by access bursts or stale frequencies. The continuous utility model integrates recency decay, frequency, size, and inter-arrival reuse simultaneously.
2. *SLA and Priority Awareness:* By assigning positive weights to $w_"prio"$ ($1.5$--$2.4$) and $w_"pen"$, the optimizers selectively protect High-Priority Content from eviction, directly avoiding heavy contractual penalties ($75$ vs. $15$).
3. *Intelligent Multi-Tier Admission:* Unlike standard policies that promote all hits to DRAM, the optimized $theta_"DRAM"$ threshold prevents DRAM tier thrashing by buffering lower-utility blocks in SSD.

== Algorithmic Trade-offs and Optimizer Selection
- *PSO vs. GA:* Both algorithms provide the fastest training convergence ($approx 11$ seconds) and lowest variance across random seeds. PSO's continuous velocity-vector trajectory makes it exceptionally well-suited to the smooth hyperparameter space, yielding the lowest overall SLA penalty on real cloud traces.
- *SFO and DSOA-EC for QoS-Critical Nodes:* When cloud providers prioritize high-priority SLA guarantees over raw aggregate hit count, SFO and DSOA-EC provide superior HPC retention (up to 70.11% CHR). SFO's dispersal mechanism and DSOA-EC's satiety decay prevent premature convergence to sub-optimal local regions.
- *ABC Overhead Considerations:* While ABC demonstrates strong exploratory capacity under concept drift, its tripartite evaluation cycle requires nearly $4times$ longer training time ($approx 42$ s), making it less suitable for high-frequency dynamic re-tuning.

== Practical Deployment and Overhead
The optimizer executes offline or asynchronously in a background tuning loop. During live request serving, victim eviction evaluates utility scores using randomized tournament selection ($K = 24$ candidates). Because each tournament requires only 24 dot-product evaluations, eviction latency is negligible ($< 5 mu"s"$), fitting well within cloud I/O budgets.

= Conclusion & Future Scope

This study presented an experimental comparative analysis of nature-inspired optimization algorithms for multi-tier cache replacement in cloud computing. Benchmarked across 10 algorithms under a strict zero-leakage 10k/40k/40k protocol, our results demonstrate:
1. Nature-inspired optimizers substantially outperform classical policies, boosting hit ratios by up to *22.17 percentage points* over LRU and reducing SLA penalty costs by more than *50%*.
2. *PSO* and *HO* delivered the highest holdout CHR (*67.62%*) on real cloud traces, while *SFO* and *DSOA-EC* achieved the highest high-priority retention (*70.11%* and *70.05%*).
3. *GA* and *PSO* offered the best operational balance, converging in under 12 seconds with low variance across seeds ($sigma <= 0.0006$).
4. Cross-workload evaluation verified that policies trained on non-stationary traces transfer effectively to stationary distributions without retraining.

Future work includes exploring lightweight online sliding-window re-tuning and extending the model to heterogeneous CXL memory pools in multi-tenant cloud environments.

#v(0.6em)
#set text(size: 8pt)
= References

#set par(justify: true, leading: 0.42em)

[1] L. Ait-Oucheggou, S. Rubini, A. Battou, and J. Boukhobza, "QM-ARC: QoS-aware Multi-tier Adaptive Cache Replacement Strategy," _Future Generation Computer Systems_, vol. 163, p. 107548, 2025.

[2] N. Megiddo and D. S. Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache," in _Proc. USENIX Conf. on File and Storage Technologies (FAST)_, pp. 115--130, 2003.

[3] L. A. Belady, "A study of replacement algorithms for a virtual-storage computer," _IBM Systems Journal_, vol. 5, no. 2, pp. 78--101, 1966.

[4] J. H. Holland, _Adaptation in Natural and Artificial Systems_, MIT Press, 1992.

[5] J. Kennedy and R. Eberhart, "Particle swarm optimization," in _Proc. IEEE Int. Conf. on Neural Networks (ICNN)_, pp. 1942--1948, 1995.

[6] D. Karaboga and B. Basturk, "A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm," _Journal of Global Optimization_, vol. 39, pp. 459--471, 2007.

[7] S. Mirjalili and A. Lewis, "The whale optimization algorithm," _Advances in Engineering Software_, vol. 95, pp. 51--67, 2016.

[8] M.-C. Su, C.-H. Hsieh, and Y.-X. Zhao, "A new metaheuristic optimization algorithm: Dove swarm optimization," _IEEE Access_, vol. 10, pp. 62153--62168, 2022.

[9] M. H. Amiri et al., "Hippopotamus optimization algorithm: a novel nature-inspired metaheuristic," _Scientific Reports_, vol. 14, 2024.

[10] Storage Networking Industry Association (SNIA), "IOTTA Repository: Cloud Block and Object Storage Traces (cluster07)," http://iotta.snia.org/traces, 2026.
