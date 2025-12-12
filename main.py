import numpy as np
import kmeans
import common
import em
import matplotlib.pyplot as plt
from common import init

def comprehensive_netflix_analysis():
    """Run EM and evaluate predictions"""
    
    X_incomplete = np.loadtxt('netflix_incomplete.txt')
    X_gold = np.loadtxt('netflix_complete.txt')
    seeds = [0, 1, 2, 3, 4]
    
    # Find best K=12 mixture
    print("Finding best K=12 mixture...")
    best_LL_12 = -np.inf
    best_mixture_12 = None
    
    for seed in seeds:
        mixture, post = init(X_incomplete, 12, seed)
        mixture, post, LL = em.run(X_incomplete, mixture, post)
        
        if LL > best_LL_12:
            best_LL_12 = LL
            best_mixture_12 = mixture
    
    print(f"Best LL for K=12: {best_LL_12:.4f}")
    
    # Evaluate predictions
    print("\nEvaluating prediction accuracy...")
    X_pred = em.fill_matrix(X_incomplete, best_mixture_12)
    rmse_score = common.rmse(X_gold, X_pred)
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    print(f"Best log-likelihood (K=12): {best_LL_12:.4f}")
    print(f"Prediction RMSE: {rmse_score:.4f}")
    
    return best_LL_12, rmse_score, best_mixture_12

# Run it
if __name__ == "__main__":
    ll_k12, rmse, best_mixture = comprehensive_netflix_analysis()