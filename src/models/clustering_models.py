import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class MarketClusterAnalysis:
    """Market clustering and segmentation analysis"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.models = {
            'kmeans': KMeans(random_state=42),
            'dbscan': DBSCAN(),
            'hierarchical': AgglomerativeClustering()
        }
        self.fitted_models = {}
        self.cluster_results = {}

    def prepare_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for market clustering"""
        # market-level aggregation
        market_features = df.groupby(['market', 'admin1', 'latitude', 'longitude']).agg({
            'price': ['mean', 'std', 'min', 'max', 'count'],
            'usdprice': ['mean', 'std'],
            'commodity': 'nunique'
        }).reset_index()

        #flatten column names
        market_features.columns = [
            'market', 'admin1', 'latitude', 'longitude',
            'price_mean', 'price_std', 'price_min', 'price_max', 'price_count', 'usdprice_mean', 'usdprice_mean', 'usd_price_std', 'commodity_count'
        ]

        #create additional features
        market_features['price_range'] = market_features['price_max'] - market_features['price_min']
        market_features['price_cv'] = market_features['price_std'] / market_features['price_mean']
        market_features['avg_commodities_per_record'] = market_features['commodity_count'] / market_features['price_count']

        #handle missing values
        market_features = market_features.fillna(0)

        return market_features
    
    def find_optimal_cluster(self, X: np.ndarray, max_cluster: int = 10) -> Dict[str, Any]:
        """Find optimal number of clusters using various metrics"""
        inertias = []
        silhouette_scores = []
        ch_scores = []
        k_range = range(2, min(max_cluster + 1, len(X)))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(X)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, cluster_labels))
            ch_scores.append(calinski_harabasz_score(X, cluster_labels))

        #find optimal k using silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]

        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'ch_scores': ch_scores,
            'optimal_k': optimal_k
        }
    
    def cluster_markets(self, df: pd.DataFrame, method: str = 'kmeans', n_cluster: int = None) -> Dict[str, Any]:
        """Perform market clustering analysis"""
        try:
            #prepare features
            market_features = self.prepare_market_features(df)
            #select numerical features for clustering
            feature_cols = ['price_mean', 'price_std', 'price_range', 'price_cv', 'usdprice_mean', 'commodity_count', 'price_count']

            X = market_features[feature_cols].values

            #scale features
            X_scaled = self.scaler.fit_transform(X)

            #find optimal clusters if not specified
            if n_cluster is None and method == 'kmeans':
                cluster_analysis = self.find_optimal_cluster(X_scaled)
                n_clusters = cluster_analysis['optimal_k']
            else:
                cluster_analysis = None
                n_clusters = n_clusters or 5
            
            #perform clustering
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = model.fit_predict(X_scaled)
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = model.fit_predict(X_scaled)
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            elif method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = model.fit_predict(X_scaled)
            
            #store fitted model
            self.fitted_models[method] = model

            #add cluster labels to market features
            market_features['cluster'] = cluster_labels

            #pca for visualization
            X_pca = self.pca.fit_transform(X_scaled)
            market_features['pca1'] = X_pca[:, 0]
            market_features['pca2'] = X_pca[:, 1]

            #calculate cluster statistics
            cluster_stats = market_features.groupby('cluster').agg({
                'price_mean': ['mean', 'std', 'count'],
                'price_cv': 'mean',
                'commodity_count': 'mean',
                'latitude': 'mean',
                'longitutd': 'mean'
            }).round(2)

            #silhouette score
            if len(set(cluster_labels)) > 1:
                silhouette_avg =  silhouette_score(X_scaled, cluster_labels)
            else:
                silhouette_avg = 0
            
            results = {
                'market_features': market_features,
                'cluster_labels': cluster_labels,
                'n_clusters': n_clusters,
                'cluster_stats': cluster_stats,
                'silhouette_score': silhouette_avg,
                'feature_columns': feature_cols,
                'method': method
            }

            if cluster_analysis:
                results['optimization'] = cluster_analysis

            self.cluster_results[method] = results

            logger.info(f'Market clustering complete. Method: {method}, Clusters: {n_clusters}')

            return results
        except Exception as e:
            logger.error(f'Error in market clustering: {e}')
            raise
    
    def analyze_commodity_cluster(self, df: pd.DataFrame, method: str = 'kmeans') -> Dict[str, Any]:
        """Cluster commodities based on price patterns"""
        try:
            #prepare commodity features
            commodity_features = df.groupby('commodity').agg({
                'price': ['mean', 'std', 'min', 'max'],
                'usdprice': ['mean', 'std'],
                'admin1': 'nunique', # number of states
                'market': 'nunique' # number of markets
            }).reset_index()

            #flatten columns
            commodity_features.columns = [
                'commodity', 'price_mean', 'price_std', 'price_min', 'price_max',
                'usdprice_mean', 'usdprice_std', 'states_count', 'markets_count'
            ]

            #additional features
            commodity_features['price_range'] = commodity_features['price_max'] - commodity_features['price_min']
            commodity_features['price_cv'] = commodity_features['price_std'] / commodity_features['price_mean']

            #seasonal analysis (if date information available)
            if 'date' in df.columns:
                seasonal_data = df.copy()
                seasonal_data['month'] = pd.to_datetime(seasonal_data['date']).dt.month

                seasonal_stats = seasonal_data.groupby(['commodity', 'month'])['price'].mean().reset_index()
                seasonal_cv =  seasonal_stats.groupby('commodity')['price'].std() / seasonal_stats.groupby('commodity')['price'].mean()

                commodity_features = commodity_features.merge(
                    seasonal_cv.reset_index().rename(columns={'price': 'seasonal_cv'}),
                    on='commodity', how='left'
                )

            #select features for clustering
            feature_cols = ['price_mean', 'price_cv', 'price_range', 'states_count', 'markets_count']
            if 'seasonal_cv' in commodity_features.columns:
                feature_cols.append('seasonal_cv')
            
            X = commodity_features[feature_cols].fillna(0).values
            X_scaled = self.scaler.fit_transform(X)

            #perform clustering
            n_clusters = min(5, len(commodity_features) // 2) #adaptive cluster count

            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = model.fit_predict(X_scaled)
            commodity_features['cluster'] = cluster_labels

            #pca for visualization
            X_pca = self.pca.fit_transform(X_scaled)
            commodity_features['pca1'] = X_pca[:, 0]
            commodity_features['pca2'] = X_pca[:, 1]

            #cluster characterization
            cluster_profiles = commodity_features.groupby('cluster').agg({
                'price_mean': 'mean',
                'price_cv': 'mean',
                'states_count': 'mean',
                'markets_count': 'mean',
                'commodity': 'count'
            }).round(2)

            return {
                'commodity_features': commodity_features,
                'cluster_profiles': cluster_profiles,
                'n_clusters': n_clusters,
                'feature_columns': feature_cols
            }
        except Exception as e:
            logger.error(f'Error in commodity clustering: {e}')
            raise
        