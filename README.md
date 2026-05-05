# aba-mining-tasks-graph-embedding-kmitl-internship-2026
This is a repository that contains the experiments for ABA mining tasks (using graph embedding), jointly conducted with internship student from KMITL in April 2026.

Project Structure
ABA-MINING-TASKS-GRAPH-EMBEDDING-KMITL-INTERNSHIP-2026/
│
├── rotate_output/                    # Output folder — RotatE full dataset
│   └── negative_samples.csv         # Negative triples captured during training
│
├── rotate_pnnp_output/               # Output folder — RotatE PNNP dataset
│   └── negative_samples.csv         # Negative triples captured during training
│
├── transe_output/                    # Output folder — TransE full dataset
│   └── negative_samples.csv         # Negative triples captured during training
│
├── transe_pnnp_output/               # Output folder — TransE PNNP dataset
│   └── negative_samples.csv         # Negative triples captured during training
│
├── experiment_results.xlsx           # Auto-recorded results for all runs
├── hotel_contrary_dataset_all.csv    # Full dataset (91,714 triples)
├── hotel_contrary_dataset_PNNP.csv   # Filtered dataset (13,942 triples)
├── README.md                         # Project documentation
│
├── rotate_EXT.py                     # RotatE — full dataset (ExtendedBasicNegativeSampler)
├── rotate_pnnp_EXT.py                # RotatE — PNNP dataset (ExtendedBasicNegativeSampler)
├── rotate_pnnp.py                    # RotatE — PNNP dataset (standard basic sampler)
├── rotate.py                         # RotatE — full dataset (standard basic sampler)
├── transe_EXT.py                     # TransE — full dataset (ExtendedBasicNegativeSampler)
├── transe_pnnp_EXT.py                # TransE — PNNP dataset (ExtendedBasicNegativeSampler)
├── transe_pnnp.py                    # TransE — PNNP dataset (standard basic sampler)
└── transe.py                         # TransE — full dataset (standard basic sampler)

Datasets
Full Dataset (hotel_contrary_dataset_all.csv)

Total triples: 91,714
Source: All 4 sheet types from the original Excel files
Relations:

CONTRARY_TO — 4,776 (5.2%) — human-verified contrary pairs
NOT_CONTRARY — 86,938 (94.8%) — verified non-contrary pairs

Filtered Dataset — PNNP (hotel_contrary_dataset_PNNP.csv)

Total triples: 13,942
Source: Only Contrary(P)Body(N) and Contrary(N)Body(P) sheets
Relations:

CONTRARY_TO — 4,776 (34.3%)
NOT_CONTRARY — 9,166 (65.7%)



