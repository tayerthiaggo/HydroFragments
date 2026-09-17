# Riverscape Phase 4 Fitzroy Bridge Calibration Findings

**Run date:** 2026-09-15  
**Cases:** 100  
**Runtime seconds:** 150.4  
**Selected candidate:** baseline  
**Selected weights:** {"bare": 0.5, "green": 1.0, "line_distance": 0.5, "npv": 0.0, "terrain": 1.0, "water": 1.0}  
**bridge_max_cost_per_m:** 1  
**Threshold fallback:** True

## Method

Reaches were stratified by UpstrDArea inside a connected local AHGF neighbourhood, buffered into a section AOI (corridor_max + small halo) so corridor/channel/REM never run on the full catchment. One hundred 300-1800 m spans were sampled along AHGF linestrings on that section. Water/domain evidence was hidden on each span; routes retained terrain, FC, and line distance, with least-cost MCP cropped to each gap window. Candidate score was median one-pixel-tolerance F1 minus twice off-channel crossing rate. Baseline was retained unless a candidate improved by at least 0.02 with at least 80% routable cases. Config defaults unchanged: selected candidate was baseline and cost-cap fell back to 1.0 (<80 high-F1 routes).

## Candidate results

- baseline: score=0.794872, routable=41, median_f1=0.794872, off_channel_rate=0.000000
- without_terrain: score=0.676868, routable=41, median_f1=0.676868, off_channel_rate=0.000000
- without_water: score=0.794872, routable=41, median_f1=0.794872, off_channel_rate=0.000000
- without_bare: score=0.859198, routable=41, median_f1=0.859198, off_channel_rate=0.000000
- without_green: score=0.800274, routable=41, median_f1=0.800274, off_channel_rate=0.000000
- without_line_distance: score=0.769340, routable=41, median_f1=0.769340, off_channel_rate=0.000000
- npv_enabled: score=0.769340, routable=41, median_f1=0.769340, off_channel_rate=0.000000
- terrain_heavy: score=0.802920, routable=41, median_f1=0.802920, off_channel_rate=0.000000
- water_light: score=0.789610, routable=41, median_f1=0.789610, off_channel_rate=0.000000
- line_light: score=0.769340, routable=41, median_f1=0.769340, off_channel_rate=0.000000
