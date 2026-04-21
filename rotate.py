import pandas as pd
import torch
import time
from pathlib import Path
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

CSV_PATH = "hotel_contrary_dataset_all.csv"
OUTPUT_DIR = Path("transe_ouput")
OUTPUT_DIR.mkdir(exist_ok=True)

EMBEDDING_DIM = 100
NUM_EPOCHS = 1000
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
MARGIN = 1.0
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42
NUM_NEG = 50

print ("="*60)
print("Loading triples")
print("="*60)

df = pd.read_csv(CSV_PATH)
print(f"Total rows loaded: {len(df):,}")
print(f"Relations found: {df['relation'].value_counts().to_dict()}")

triples_array = df[['head', 'relation', 'tail']].values.astype(str)

print ("="*60)
print("Creating TriplesFactory")
print("="*60)

tf = TriplesFactory.from_labeled_triples(triples=triples_array)
print(f"Unique entities : {tf.num_entities:,}")
print(f"Unique relations : {tf.num_relations}")

training, validation, testing = tf.split(
    ratios=[TRAIN_RATIO, VAL_RATIO],
    random_state=RANDOM_SEED,
)
print(f"Train: {training.num_triples:,}")
print(f"Val  : {validation.num_triples:,}")
print(f"Test : {testing.num_triples:,}")

print ("="*60)
print("Training RotatE")
print ("="*60)

start_time = time.time()

result = pipeline(
    training=training,
    validation=validation,
    testing=testing,

    model="RotatE",
    model_kwargs=dict(
        embedding_dim=EMBEDDING_DIM,
        ),

    loss="NSSALoss",
    loss_kwargs=dict(
        margin=MARGIN,
        adversarial_temperature=0.5,
    ),

    negative_sampler="basic",
    negative_sampler_kwargs=dict(
        num_negs_per_pos=NUM_NEG,
        corruption_scheme=["head"],
    ),

    optimizer="Adam",
    optimizer_kwargs=dict(lr=LEARNING_RATE),

    training_loop="sLCWA",
    training_kwargs=dict(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    ),

    stopper="early",
    stopper_kwargs=dict(
        metric="hits_at_10",
        patience=10,
        relative_delta=0.002,
        frequency=10,
    ),

    evaluator="RankBasedEvaluator",
    evaluator_kwargs=dict(filtered=True),

    random_seed=RANDOM_SEED,

    device="cuda" if torch.cuda.is_available() else "cpu",
)

end_time = time.time()
elapsed = end_time - start_time
print(f"\nTraining completed in: {int(elapsed//3600)}h {int((elapsed%3600)//60)}m {int(elapsed%60)}s")

print ("="*60)
print("Evaluation Results")
print("="*60)

metrics = result.metric_results.to_dict()
r = metrics["both"]["realistic"]

print(f"MRR         : {r['inverse_harmonic_mean_rank']:.4f}")
print(f"Hits@1      : {r['hits_at_1']:.4f}")
print(f"Hits@3      : {r['hits_at_3']:.4f}")
print(f"Hits@10     : {r['hits_at_10']:.4f}")
print(f"Mean Rank   : {r['arithmetic_mean_rank']:.4f}")

'''
print("Saving outputs")
print("=" * 60)

result.save_to_directory(OUTPUT_DIR / "pipeline_result")
print(f"  Model saved to: {OUTPUT_DIR / 'pipeline_result'}")
'''
