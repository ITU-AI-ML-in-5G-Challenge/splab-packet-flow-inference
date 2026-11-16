# Exact feature inventories (selected v11 release)

Counts precede one-hot expansion. Task 1 uses the61-column main ET,63-column full/prefix CatBoost,17-column body HGB and fixed-kernel ridge. Task 2 uses the33-column content method, shared latent-time correction, class/protocol support, downstream-size interpolation and strictly past capture RFs.

## task1_static

### Main ET (`compact`): 61 columns

| # | Column | Definition |
|---:|---|---|
| 1 | `duration` | Elapsed time from the first to last packet (seconds). |
| 2 | `packet_count_up` | Number of upstream packets. |
| 3 | `packet_count_down` | Number of downstream packets. |
| 4 | `bytes_up` | Sum of upstream packet sizes (bytes). |
| 5 | `bytes_down` | Sum of downstream packet sizes (bytes). |
| 6 | `mean_pkt_size_up` | Mean upstream packet size. |
| 7 | `mean_pkt_size_down` | Mean downstream packet size. |
| 8 | `std_pkt_size_up` | Population standard deviation of upstream size. |
| 9 | `std_pkt_size_down` | Population standard deviation of downstream size. |
| 10 | `mean_iat` | Mean interarrival gap (seconds). |
| 11 | `std_iat` | Population standard deviation of interarrival gaps. |
| 12 | `min_iat` | Minimum interarrival gap. |
| 13 | `max_iat` | Maximum interarrival gap. |
| 14 | `burst_count` | Released burst count. |
| 15 | `shape_pl_cv_up` | Upstream size standard deviation / mean. |
| 16 | `shape_pl_cv_down` | Downstream size standard deviation / mean. |
| 17 | `shape_packet_up_share` | Upstream count / total count. |
| 18 | `shape_byte_up_share` | Upstream bytes / total bytes. |
| 19 | `shape_iat_cv` | IAT standard deviation / IAT mean. |
| 20 | `shape_iat_max_duration` | Maximum gap / duration. |
| 21 | `active_iat_mean` | (duration - max_iat) / (total packet count - 2). |
| 22 | `active_seconds_per_kib` | (duration - max_iat) / (total bytes / 1024). |
| 23 | `trimmed_iat_cv` | Population IAT coefficient of variation after removing one maximum gap; exact moments and validity checks. |
| 24 | `transport` | TCP=0, UDP=1. |
| 25 | `p_duration` | Prefix: Elapsed time from the first to last packet (seconds). |
| 26 | `p_packet_count_up` | Prefix: Number of upstream packets. |
| 27 | `p_packet_count_down` | Prefix: Number of downstream packets. |
| 28 | `p_bytes_up` | Prefix: Sum of upstream packet sizes (bytes). |
| 29 | `p_bytes_down` | Prefix: Sum of downstream packet sizes (bytes). |
| 30 | `p_mean_pkt_size_up` | Prefix: Mean upstream packet size. |
| 31 | `p_mean_pkt_size_down` | Prefix: Mean downstream packet size. |
| 32 | `p_std_pkt_size_up` | Prefix: Population standard deviation of upstream size. |
| 33 | `p_std_pkt_size_down` | Prefix: Population standard deviation of downstream size. |
| 34 | `p_mean_iat` | Prefix: Mean interarrival gap (seconds). |
| 35 | `p_std_iat` | Prefix: Population standard deviation of interarrival gaps. |
| 36 | `p_min_iat` | Prefix: Minimum interarrival gap. |
| 37 | `p_max_iat` | Prefix: Maximum interarrival gap. |
| 38 | `p_burst_count` | Prefix: Released burst count. |
| 39 | `p_up_down_packet_ratio` | Prefix: Released upstream/downstream packet-count ratio. |
| 40 | `p_up_down_byte_ratio` | Prefix: Released upstream/downstream byte ratio. |
| 41 | `p_up_max_lower` | Prefix up: mu + sigma / sqrt(n-1), with n=1 handled separately. |
| 42 | `p_up_max_upper` | Prefix up: mu + sigma * sqrt(n-1). |
| 43 | `p_up_min_lower` | Prefix up: max(mu - sigma * sqrt(n-1), 0). |
| 44 | `p_up_min_upper` | Prefix up: max(mu - sigma / sqrt(n-1), 0), with n=1 handled separately. |
| 45 | `p_up_exact_extrema_available` | Prefix up: n is 1 or 2. |
| 46 | `p_up_exact_min` | Prefix up: mu-sigma when n is 1 or 2; otherwise missing. |
| 47 | `p_up_exact_max` | Prefix up: mu+sigma when n is 1 or 2; otherwise missing. |
| 48 | `p_down_max_lower` | Prefix down: mu + sigma / sqrt(n-1), with n=1 handled separately. |
| 49 | `p_down_max_upper` | Prefix down: mu + sigma * sqrt(n-1). |
| 50 | `p_down_min_lower` | Prefix down: max(mu - sigma * sqrt(n-1), 0). |
| 51 | `p_down_min_upper` | Prefix down: max(mu - sigma / sqrt(n-1), 0), with n=1 handled separately. |
| 52 | `p_down_exact_extrema_available` | Prefix down: n is 1 or 2. |
| 53 | `p_down_exact_min` | Prefix down: mu-sigma when n is 1 or 2; otherwise missing. |
| 54 | `p_down_exact_max` | Prefix down: mu+sigma when n is 1 or 2; otherwise missing. |
| 55 | `fp_up_bytes` | Protocol, prefix upstream count and integer upstream byte sum as a string key. |
| 56 | `fp_up_max_bound` | Protocol and rounded upper bound on the maximum prefix upstream size as a string key. |
| 57 | `r_byte_up_share` | Residual upstream bytes / residual total bytes. |
| 58 | `r_up_mean_size` | Residual upstream bytes / residual upstream count. |
| 59 | `r_down_mean_size` | Residual downstream bytes / residual downstream count. |
| 60 | `r_after5_seconds` | Full duration - prefix duration. |
| 61 | `p_byte_share` | Prefix total bytes / full total bytes. |

### Full-flow/prefix CatBoost (`full_cat`): 63 columns

| # | Column | Definition |
|---:|---|---|
| 1 | `duration` | Elapsed time from the first to last packet (seconds). |
| 2 | `packet_count_up` | Number of upstream packets. |
| 3 | `packet_count_down` | Number of downstream packets. |
| 4 | `bytes_up` | Sum of upstream packet sizes (bytes). |
| 5 | `bytes_down` | Sum of downstream packet sizes (bytes). |
| 6 | `mean_pkt_size_up` | Mean upstream packet size. |
| 7 | `mean_pkt_size_down` | Mean downstream packet size. |
| 8 | `std_pkt_size_up` | Population standard deviation of upstream size. |
| 9 | `std_pkt_size_down` | Population standard deviation of downstream size. |
| 10 | `mean_iat` | Mean interarrival gap (seconds). |
| 11 | `std_iat` | Population standard deviation of interarrival gaps. |
| 12 | `min_iat` | Minimum interarrival gap. |
| 13 | `max_iat` | Maximum interarrival gap. |
| 14 | `burst_count` | Released burst count. |
| 15 | `shape_pl_cv_up` | Upstream size standard deviation / mean. |
| 16 | `shape_pl_cv_down` | Downstream size standard deviation / mean. |
| 17 | `shape_packet_up_share` | Upstream count / total count. |
| 18 | `shape_byte_up_share` | Upstream bytes / total bytes. |
| 19 | `shape_iat_cv` | IAT standard deviation / IAT mean. |
| 20 | `shape_iat_max_duration` | Maximum gap / duration. |
| 21 | `active_iat_mean` | (duration - max_iat) / (total packet count - 2). |
| 22 | `active_seconds_per_kib` | (duration - max_iat) / (total bytes / 1024). |
| 23 | `trimmed_iat_cv` | Population IAT coefficient of variation after removing one maximum gap; exact moments and validity checks. |
| 24 | `transport` | TCP=0, UDP=1. |
| 25 | `p_duration` | Prefix: Elapsed time from the first to last packet (seconds). |
| 26 | `p_packet_count_up` | Prefix: Number of upstream packets. |
| 27 | `p_packet_count_down` | Prefix: Number of downstream packets. |
| 28 | `p_bytes_up` | Prefix: Sum of upstream packet sizes (bytes). |
| 29 | `p_bytes_down` | Prefix: Sum of downstream packet sizes (bytes). |
| 30 | `p_mean_pkt_size_up` | Prefix: Mean upstream packet size. |
| 31 | `p_mean_pkt_size_down` | Prefix: Mean downstream packet size. |
| 32 | `p_std_pkt_size_up` | Prefix: Population standard deviation of upstream size. |
| 33 | `p_std_pkt_size_down` | Prefix: Population standard deviation of downstream size. |
| 34 | `p_mean_iat` | Prefix: Mean interarrival gap (seconds). |
| 35 | `p_std_iat` | Prefix: Population standard deviation of interarrival gaps. |
| 36 | `p_min_iat` | Prefix: Minimum interarrival gap. |
| 37 | `p_max_iat` | Prefix: Maximum interarrival gap. |
| 38 | `p_burst_count` | Prefix: Released burst count. |
| 39 | `p_up_down_packet_ratio` | Prefix: Released upstream/downstream packet-count ratio. |
| 40 | `p_up_down_byte_ratio` | Prefix: Released upstream/downstream byte ratio. |
| 41 | `p_up_max_lower` | Prefix up: mu + sigma / sqrt(n-1), with n=1 handled separately. |
| 42 | `p_up_max_upper` | Prefix up: mu + sigma * sqrt(n-1). |
| 43 | `p_up_min_lower` | Prefix up: max(mu - sigma * sqrt(n-1), 0). |
| 44 | `p_up_min_upper` | Prefix up: max(mu - sigma / sqrt(n-1), 0), with n=1 handled separately. |
| 45 | `p_up_exact_extrema_available` | Prefix up: n is 1 or 2. |
| 46 | `p_up_exact_min` | Prefix up: mu-sigma when n is 1 or 2; otherwise missing. |
| 47 | `p_up_exact_max` | Prefix up: mu+sigma when n is 1 or 2; otherwise missing. |
| 48 | `p_down_max_lower` | Prefix down: mu + sigma / sqrt(n-1), with n=1 handled separately. |
| 49 | `p_down_max_upper` | Prefix down: mu + sigma * sqrt(n-1). |
| 50 | `p_down_min_lower` | Prefix down: max(mu - sigma * sqrt(n-1), 0). |
| 51 | `p_down_min_upper` | Prefix down: max(mu - sigma / sqrt(n-1), 0), with n=1 handled separately. |
| 52 | `p_down_exact_extrema_available` | Prefix down: n is 1 or 2. |
| 53 | `p_down_exact_min` | Prefix down: mu-sigma when n is 1 or 2; otherwise missing. |
| 54 | `p_down_exact_max` | Prefix down: mu+sigma when n is 1 or 2; otherwise missing. |
| 55 | `fp_up_bytes` | Protocol, prefix upstream count and integer upstream byte sum as a string key. |
| 56 | `fp_up_max_bound` | Protocol and rounded upper bound on the maximum prefix upstream size as a string key. |
| 57 | `r_byte_up_share` | Residual upstream bytes / residual total bytes. |
| 58 | `r_up_mean_size` | Residual upstream bytes / residual upstream count. |
| 59 | `r_down_mean_size` | Residual downstream bytes / residual downstream count. |
| 60 | `r_after5_seconds` | Full duration - prefix duration. |
| 61 | `p_byte_share` | Prefix total bytes / full total bytes. |
| 62 | `struct_tcp_m` | Protocol and fitted TCP residual-size structural category. |
| 63 | `struct_up_vocab` | Protocol and maximum/middle sizes of the fitting-vocabulary-selected multiset. |

### Body HGB (`hgb`): 17 columns

| # | Column | Definition |
|---:|---|---|
| 1 | `f_duration` | Full-flow duration. |
| 2 | `f_bursts_per_second` | Full-flow burst count / duration. |
| 3 | `f_packets_per_burst` | Full-flow total packets / burst count. |
| 4 | `f_down_mean` | Full-flow downstream mean size, missing when absent. |
| 5 | `f_down_std` | Full-flow downstream size standard deviation, missing when absent. |
| 6 | `f_up_byte_share` | Full-flow upstream bytes / total bytes. |
| 7 | `r_duration` | Full duration - prefix duration. |
| 8 | `r_burst_difference_per_second` | (full burst count - prefix burst count) / residual duration. |
| 9 | `r_packets` | Full packet count - prefix packet count. |
| 10 | `r_down_mean` | Residual downstream mean size. |
| 11 | `r_down_std` | Residual downstream population size standard deviation. |
| 12 | `r_up_byte_share` | Residual upstream bytes / residual total bytes. |
| 13 | `f_max_iat` | Full-flow maximum interarrival gap. |
| 14 | `f_iat_cv` | Full-flow IAT standard deviation / mean. |
| 15 | `f_max_gap_fraction` | Full-flow maximum gap / duration. |
| 16 | `r_down_cv` | Residual downstream size standard deviation / mean. |
| 17 | `transport` | TCP=0, UDP=1. |

### Ranker main-design fields (`ranker_physical`): 58 columns

| # | Column | Definition |
|---:|---|---|
| 1 | `duration` | Elapsed time from the first to last packet (seconds). |
| 2 | `packet_count_up` | Number of upstream packets. |
| 3 | `packet_count_down` | Number of downstream packets. |
| 4 | `bytes_up` | Sum of upstream packet sizes (bytes). |
| 5 | `bytes_down` | Sum of downstream packet sizes (bytes). |
| 6 | `mean_pkt_size_up` | Mean upstream packet size. |
| 7 | `mean_pkt_size_down` | Mean downstream packet size. |
| 8 | `std_pkt_size_up` | Population standard deviation of upstream size. |
| 9 | `std_pkt_size_down` | Population standard deviation of downstream size. |
| 10 | `mean_iat` | Mean interarrival gap (seconds). |
| 11 | `std_iat` | Population standard deviation of interarrival gaps. |
| 12 | `min_iat` | Minimum interarrival gap. |
| 13 | `max_iat` | Maximum interarrival gap. |
| 14 | `burst_count` | Released burst count. |
| 15 | `shape_pl_cv_up` | Upstream size standard deviation / mean. |
| 16 | `shape_pl_cv_down` | Downstream size standard deviation / mean. |
| 17 | `shape_packet_up_share` | Upstream count / total count. |
| 18 | `shape_byte_up_share` | Upstream bytes / total bytes. |
| 19 | `shape_iat_cv` | IAT standard deviation / IAT mean. |
| 20 | `shape_iat_max_duration` | Maximum gap / duration. |
| 21 | `active_iat_mean` | (duration - max_iat) / (total packet count - 2). |
| 22 | `active_seconds_per_kib` | (duration - max_iat) / (total bytes / 1024). |
| 23 | `trimmed_iat_cv` | Population IAT coefficient of variation after removing one maximum gap; exact moments and validity checks. |
| 24 | `p_duration` | Prefix: Elapsed time from the first to last packet (seconds). |
| 25 | `p_packet_count_up` | Prefix: Number of upstream packets. |
| 26 | `p_packet_count_down` | Prefix: Number of downstream packets. |
| 27 | `p_bytes_up` | Prefix: Sum of upstream packet sizes (bytes). |
| 28 | `p_bytes_down` | Prefix: Sum of downstream packet sizes (bytes). |
| 29 | `p_mean_pkt_size_up` | Prefix: Mean upstream packet size. |
| 30 | `p_mean_pkt_size_down` | Prefix: Mean downstream packet size. |
| 31 | `p_std_pkt_size_up` | Prefix: Population standard deviation of upstream size. |
| 32 | `p_std_pkt_size_down` | Prefix: Population standard deviation of downstream size. |
| 33 | `p_mean_iat` | Prefix: Mean interarrival gap (seconds). |
| 34 | `p_std_iat` | Prefix: Population standard deviation of interarrival gaps. |
| 35 | `p_min_iat` | Prefix: Minimum interarrival gap. |
| 36 | `p_max_iat` | Prefix: Maximum interarrival gap. |
| 37 | `p_burst_count` | Prefix: Released burst count. |
| 38 | `p_up_down_packet_ratio` | Prefix: Released upstream/downstream packet-count ratio. |
| 39 | `p_up_down_byte_ratio` | Prefix: Released upstream/downstream byte ratio. |
| 40 | `p_up_max_lower` | Prefix up: mu + sigma / sqrt(n-1), with n=1 handled separately. |
| 41 | `p_up_max_upper` | Prefix up: mu + sigma * sqrt(n-1). |
| 42 | `p_up_min_lower` | Prefix up: max(mu - sigma * sqrt(n-1), 0). |
| 43 | `p_up_min_upper` | Prefix up: max(mu - sigma / sqrt(n-1), 0), with n=1 handled separately. |
| 44 | `p_up_exact_extrema_available` | Prefix up: n is 1 or 2. |
| 45 | `p_up_exact_min` | Prefix up: mu-sigma when n is 1 or 2; otherwise missing. |
| 46 | `p_up_exact_max` | Prefix up: mu+sigma when n is 1 or 2; otherwise missing. |
| 47 | `p_down_max_lower` | Prefix down: mu + sigma / sqrt(n-1), with n=1 handled separately. |
| 48 | `p_down_max_upper` | Prefix down: mu + sigma * sqrt(n-1). |
| 49 | `p_down_min_lower` | Prefix down: max(mu - sigma * sqrt(n-1), 0). |
| 50 | `p_down_min_upper` | Prefix down: max(mu - sigma / sqrt(n-1), 0), with n=1 handled separately. |
| 51 | `p_down_exact_extrema_available` | Prefix down: n is 1 or 2. |
| 52 | `p_down_exact_min` | Prefix down: mu-sigma when n is 1 or 2; otherwise missing. |
| 53 | `p_down_exact_max` | Prefix down: mu+sigma when n is 1 or 2; otherwise missing. |
| 54 | `r_byte_up_share` | Residual upstream bytes / residual total bytes. |
| 55 | `r_up_mean_size` | Residual upstream bytes / residual upstream count. |
| 56 | `r_down_mean_size` | Residual downstream bytes / residual downstream count. |
| 57 | `r_after5_seconds` | Full duration - prefix duration. |
| 58 | `p_byte_share` | Prefix total bytes / full total bytes. |

### Ranker body fields (`ranker_body`): 37 columns

| # | Column | Definition |
|---:|---|---|
| 1 | `transport` | TCP=0, UDP=1. |
| 2 | `p_duration` | Prefix: Elapsed time from the first to last packet (seconds). |
| 3 | `p_nup` | Prefix: Row-local prefix statistic under the named construction. |
| 4 | `p_bup` | Prefix: Row-local prefix statistic under the named construction. |
| 5 | `p_bdown` | Prefix: Row-local prefix statistic under the named construction. |
| 6 | `p_iat_mean` | Prefix: Row-local prefix statistic under the named construction. |
| 7 | `p_iat_std` | Prefix: Row-local prefix statistic under the named construction. |
| 8 | `p_iat_max` | Prefix: Row-local prefix statistic under the named construction. |
| 9 | `p_bursts` | Prefix: Row-local prefix statistic under the named construction. |
| 10 | `p_mean_up` | Prefix: Row-local prefix statistic under the named construction. |
| 11 | `p_std_up` | Prefix: Row-local prefix statistic under the named construction. |
| 12 | `p_up_available` | Prefix: Row-local prefix statistic under the named construction. |
| 13 | `p_cv_up` | Prefix: Row-local prefix statistic under the named construction. |
| 14 | `p_mean_down` | Prefix: Row-local prefix statistic under the named construction. |
| 15 | `p_std_down` | Prefix: Row-local prefix statistic under the named construction. |
| 16 | `p_down_available` | Prefix: Row-local prefix statistic under the named construction. |
| 17 | `p_cv_down` | Prefix: Row-local prefix statistic under the named construction. |
| 18 | `p_up_byte_share` | Prefix: Row-local prefix statistic under the named construction. |
| 19 | `p_iat_cv` | Prefix: Row-local prefix statistic under the named construction. |
| 20 | `p_max_gap_fraction` | Prefix: Row-local prefix statistic under the named construction. |
| 21 | `p_mean_contrast` | Prefix: Row-local prefix statistic under the named construction. |
| 22 | `f_duration` | Full-flow duration. |
| 23 | `f_bursts_per_second` | Full-flow burst count / duration. |
| 24 | `f_packets_per_burst` | Full-flow total packets / burst count. |
| 25 | `f_down_mean` | Full-flow downstream mean size, missing when absent. |
| 26 | `f_down_std` | Full-flow downstream size standard deviation, missing when absent. |
| 27 | `f_up_byte_share` | Full-flow upstream bytes / total bytes. |
| 28 | `r_duration` | Full duration - prefix duration. |
| 29 | `r_burst_difference_per_second` | (full burst count - prefix burst count) / residual duration. |
| 30 | `r_packets` | Full packet count - prefix packet count. |
| 31 | `r_down_mean` | Residual downstream mean size. |
| 32 | `r_down_std` | Residual downstream population size standard deviation. |
| 33 | `r_up_byte_share` | Residual upstream bytes / residual total bytes. |
| 34 | `f_max_iat` | Full-flow maximum interarrival gap. |
| 35 | `f_iat_cv` | Full-flow IAT standard deviation / mean. |
| 36 | `f_max_gap_fraction` | Full-flow maximum gap / duration. |
| 37 | `r_down_cv` | Residual downstream size standard deviation / mean. |

## task2_realtime_prefix5

### ET: 33 columns

| # | Column | Definition |
|---:|---|---|
| 1 | `duration` | Elapsed time from the first to last packet (seconds). |
| 2 | `transport` | TCP=0, UDP=1. |
| 3 | `packet_count_up` | Number of upstream packets. |
| 4 | `packet_count_down` | Number of downstream packets. |
| 5 | `bytes_up` | Sum of upstream packet sizes (bytes). |
| 6 | `bytes_down` | Sum of downstream packet sizes (bytes). |
| 7 | `mean_pkt_size_up` | Mean upstream packet size. |
| 8 | `mean_pkt_size_down` | Mean downstream packet size. |
| 9 | `std_pkt_size_up` | Population standard deviation of upstream size. |
| 10 | `std_pkt_size_down` | Population standard deviation of downstream size. |
| 11 | `mean_iat` | Mean interarrival gap (seconds). |
| 12 | `std_iat` | Population standard deviation of interarrival gaps. |
| 13 | `min_iat` | Minimum interarrival gap. |
| 14 | `max_iat` | Maximum interarrival gap. |
| 15 | `burst_count` | Released burst count. |
| 16 | `up_down_packet_ratio` | Released upstream/downstream packet-count ratio. |
| 17 | `up_down_byte_ratio` | Released upstream/downstream byte ratio. |
| 18 | `fp_up_bytes` | Protocol, prefix upstream count and integer upstream byte sum as a string key. |
| 19 | `fp_up_max_bound` | Protocol and rounded upper bound on the maximum prefix upstream size as a string key. |
| 20 | `p_up_max_lower` | Prefix up: mu + sigma / sqrt(n-1), with n=1 handled separately. |
| 21 | `p_up_max_upper` | Prefix up: mu + sigma * sqrt(n-1). |
| 22 | `p_up_min_lower` | Prefix up: max(mu - sigma * sqrt(n-1), 0). |
| 23 | `p_up_min_upper` | Prefix up: max(mu - sigma / sqrt(n-1), 0), with n=1 handled separately. |
| 24 | `p_up_exact_extrema_available` | Prefix up: n is 1 or 2. |
| 25 | `p_up_exact_min` | Prefix up: mu-sigma when n is 1 or 2; otherwise missing. |
| 26 | `p_up_exact_max` | Prefix up: mu+sigma when n is 1 or 2; otherwise missing. |
| 27 | `p_down_max_lower` | Prefix down: mu + sigma / sqrt(n-1), with n=1 handled separately. |
| 28 | `p_down_max_upper` | Prefix down: mu + sigma * sqrt(n-1). |
| 29 | `p_down_min_lower` | Prefix down: max(mu - sigma * sqrt(n-1), 0). |
| 30 | `p_down_min_upper` | Prefix down: max(mu - sigma / sqrt(n-1), 0), with n=1 handled separately. |
| 31 | `p_down_exact_extrema_available` | Prefix down: n is 1 or 2. |
| 32 | `p_down_exact_min` | Prefix down: mu-sigma when n is 1 or 2; otherwise missing. |
| 33 | `p_down_exact_max` | Prefix down: mu+sigma when n is 1 or 2; otherwise missing. |

### CatBoost: 33 columns

| # | Column | Definition |
|---:|---|---|
| 1 | `duration` | Elapsed time from the first to last packet (seconds). |
| 2 | `transport` | TCP=0, UDP=1. |
| 3 | `packet_count_up` | Number of upstream packets. |
| 4 | `packet_count_down` | Number of downstream packets. |
| 5 | `bytes_up` | Sum of upstream packet sizes (bytes). |
| 6 | `bytes_down` | Sum of downstream packet sizes (bytes). |
| 7 | `mean_pkt_size_up` | Mean upstream packet size. |
| 8 | `mean_pkt_size_down` | Mean downstream packet size. |
| 9 | `std_pkt_size_up` | Population standard deviation of upstream size. |
| 10 | `std_pkt_size_down` | Population standard deviation of downstream size. |
| 11 | `mean_iat` | Mean interarrival gap (seconds). |
| 12 | `std_iat` | Population standard deviation of interarrival gaps. |
| 13 | `min_iat` | Minimum interarrival gap. |
| 14 | `max_iat` | Maximum interarrival gap. |
| 15 | `burst_count` | Released burst count. |
| 16 | `up_down_packet_ratio` | Released upstream/downstream packet-count ratio. |
| 17 | `up_down_byte_ratio` | Released upstream/downstream byte ratio. |
| 18 | `fp_up_bytes` | Protocol, prefix upstream count and integer upstream byte sum as a string key. |
| 19 | `fp_up_max_bound` | Protocol and rounded upper bound on the maximum prefix upstream size as a string key. |
| 20 | `p_up_max_lower` | Prefix up: mu + sigma / sqrt(n-1), with n=1 handled separately. |
| 21 | `p_up_max_upper` | Prefix up: mu + sigma * sqrt(n-1). |
| 22 | `p_up_min_lower` | Prefix up: max(mu - sigma * sqrt(n-1), 0). |
| 23 | `p_up_min_upper` | Prefix up: max(mu - sigma / sqrt(n-1), 0), with n=1 handled separately. |
| 24 | `p_up_exact_extrema_available` | Prefix up: n is 1 or 2. |
| 25 | `p_up_exact_min` | Prefix up: mu-sigma when n is 1 or 2; otherwise missing. |
| 26 | `p_up_exact_max` | Prefix up: mu+sigma when n is 1 or 2; otherwise missing. |
| 27 | `p_down_max_lower` | Prefix down: mu + sigma / sqrt(n-1), with n=1 handled separately. |
| 28 | `p_down_max_upper` | Prefix down: mu + sigma * sqrt(n-1). |
| 29 | `p_down_min_lower` | Prefix down: max(mu - sigma * sqrt(n-1), 0). |
| 30 | `p_down_min_upper` | Prefix down: max(mu - sigma / sqrt(n-1), 0), with n=1 handled separately. |
| 31 | `p_down_exact_extrema_available` | Prefix down: n is 1 or 2. |
| 32 | `p_down_exact_min` | Prefix down: mu-sigma when n is 1 or 2; otherwise missing. |
| 33 | `p_down_exact_max` | Prefix down: mu+sigma when n is 1 or 2; otherwise missing. |

## Fixed-kernel ridge and selector inputs

Task 1 ridge uses the main ET numerical design, fitted standardization and the same protocol-masked similarities. There are five component kernels: numerical shape, fp_up_bytes, fp_up_max_bound, struct_tcp_m and struct_up_vocab. Each has fixed weight 0.2. These weights are not optimized and no SVM is fitted or stored. Class-balanced ridge uses strength 0.1. Vocabulary fitting uses training rows only.

The candidate ranker consumes252 values per flow/class candidate, including component scores, votes, geometry and190 physical/missingness values. These physical values are data features, not predictions from additional classifiers.

The ET/ridge default, temporal correction and minimum-two class/protocol guard apply to all classes. The guard is an empirically selected support restriction, not an independently established optimum. Task 2 uses ET, CatBoost, latent size likelihood, size/capture helpers and strictly earlier labeled temporal references. See the [technical report](../TECHNICAL_REPORT.md).

Eligible Task 2 capture RF queries adopt the RF argmax directly. Signature/class-count/current-class/past-reference eligibility stays in force. Confidence and earlier-prediction protection flags are omitted.
