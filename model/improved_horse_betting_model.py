
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')

class ImprovedHorseBettingModel:
    def __init__(self, random_state=42):
        self.base_model = GradientBoostingClassifier(random_state=random_state)
        self.model = None
        self.calibrator = None
        self.is_trained = False
        self.random_state = random_state
        self.price_band_thresholds = {
            '<3': 0.02,
            '3-6': 0.05,
            '6-10': 0.08,
            '10-20': 0.12,
            '>20': 0.20
        }
        self.fallback_min_prob = 0.28

    def _to_num(self, s):
        return pd.to_numeric(s, errors='coerce')

    def _canon_weight(self, df):
        if 'weight_carried' in df.columns:
            return df['weight_carried']
        if 'weight_carriered' in df.columns:
            return self._to_num(df['weight_carriered'])
        return pd.Series(np.nan, index=df.index)

    def _parse_grade_tier(self, s):
        if pd.isna(s):
            return np.nan
        x = str(s).strip().lower()
        mapping = {
            'g1': 1, 'group 1': 1,
            'g2': 2, 'group 2': 2,
            'g3': 3, 'group 3': 3,
            'listed': 4, 'lr': 4,
            'open': 5, 'open hcp': 5,
            'bm100': 9, 'bm95': 9,
            'bm90': 10, 'bm85': 11, 'bm80': 12, 'bm78': 12, 'bm75': 13, 'bm72': 13, 'bm70': 14,
            'bm68': 14, 'bm66': 15, 'bm64': 16, 'bm60': 17, 'bm58': 18, 'bm54': 19,
            'cl3': 20, 'cl2': 21, 'cl1': 22,
            'mdn': 30, 'mdns': 30, 'mdng': 30, 'mdnh': 30
        }
        for k, v in mapping.items():
            if k in x:
                return v
        if 'bm' in x:
            try:
                num = int(''.join([c for c in x if c.isdigit()]))
                return max(8, 40 - num // 2)
            except Exception:
                return 25
        return 25

    def _safe_group(self, df):
        cols = ['date','course','race']
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = ''
        out['date'] = pd.to_datetime(out['date'], errors='coerce')
        out['course'] = out['course'].astype(str).str.strip().str.lower()
        out['race'] = out['race'].astype(str).str.strip()
        return out

    def engineer_features(self, df):
        df = df.copy()
        df = self._safe_group(df)

        # Canonicals
        df['price'] = self._to_num(df.get('average_price'))
        df['log_price'] = np.log(df['price']).replace([np.inf, -np.inf], np.nan)
        df['weight'] = self._canon_weight(df)
        df['barrier'] = self._to_num(df.get('barrier'))
        df['dslr'] = self._to_num(df.get('days_since_last_run'))

        # Grade tiers and step-up
        df['grade_tier'] = df.get('grade').apply(self._parse_grade_tier)
        df['last_grade_tier'] = df.get('last_start_grade').apply(self._parse_grade_tier)
        df['step_up'] = df['grade_tier'] - df['last_grade_tier']

        # Recency bins
        bins = [-1,7,21,42,90,9999]
        labels = [0,1,2,3,4]
        df['dslr_bin'] = pd.cut(df['dslr'], bins=bins, labels=labels).astype('float')

        # Within-race ranks and percentiles
        grp_cols = ['date','course','race']
        df['field_size'] = df.groupby(grp_cols)['name'].transform('count') if 'name' in df.columns else df.groupby(grp_cols)['tab_number'].transform('count')
        for col, asc in [('price', True), ('barrier', True), ('weight', True), ('grade_tier', False)]:
            if col in df.columns:
                rank = df.groupby(grp_cols)[col].rank(method='average', ascending=asc)
                df[col + '_rank'] = rank
                df[col + '_pct'] = rank / df['field_size']

        # Simple trainer/jockey cumulative strike rate up to previous run (leakage-safe)
        for ent in ['trainer','jockey']:
            if ent in df.columns:
                tmp = df[[ent,'date','race','course']].copy()
                tmp['key'] = tmp[ent].astype(str).str.strip().str.lower()
                tmp['is_win'] = self._to_num(df.get('result'))
                tmp['is_win'] = tmp['is_win'].fillna(0)
                tmp = tmp.sort_values(['key','date'])
                tmp['runs_cum'] = tmp.groupby('key').cumcount()
                tmp['wins_cum'] = tmp.groupby('key')['is_win'].cumsum().shift(1).fillna(0)
                tmp['runs_cum'] = tmp['runs_cum'].astype(float)
                rate = np.where(tmp['runs_cum'] > 0, tmp['wins_cum'] / tmp['runs_cum'], np.nan)
                df[ent + '_form_cum'] = rate

        # Weight features
        if 'weight' in df.columns:
            df['weight_above_min'] = df['weight'] - df.groupby(grp_cols)['weight'].transform('min')

        # EV proxies
        df['implied_p'] = np.where(df['price'] > 0, 1.0 / df['price'], np.nan)
        return df

    def _feature_columns(self, df):
        candidates = [
            'log_price','price_rank_pct','implied_p',
            'grade_tier','last_grade_tier','step_up',
            'dslr','dslr_bin',
            'barrier','barrier_rank','barrier_pct',
            'field_size',
            'weight','weight_above_min','weight_rank','weight_pct',
            'trainer_form_cum','jockey_form_cum'
        ]
        cols = [c for c in candidates if c in df.columns]
        return cols

    def _time_split(self, df, y, val_frac=0.2):
        order = df['date'].sort_values(kind='mergesort')
        cutoff_idx = int(len(order) * (1.0 - val_frac))
        cutoff_date = order.iloc[cutoff_idx]
        train_idx = df['date'] <= cutoff_date
        val_idx = df['date'] > cutoff_date
        return train_idx, val_idx

    def fit(self, df):
        df2 = self.engineer_features(df)
        if 'result' in df2.columns:
            y = self._to_num(df2['result']).fillna(0).astype(int)
        elif 'win_flag' in df2.columns:
            y = self._to_num(df2['win_flag']).fillna(0).astype(int)
        else:
            raise ValueError('Training requires result or win_flag column')

        X_cols = self._feature_columns(df2)
        X = df2[X_cols].fillna(df2[X_cols].median())

        tr_idx, va_idx = self._time_split(df2, y)
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        base = self.base_model
        base.fit(X_tr, y_tr)
        self.model = base

        # Calibrate with isotonic on validation
        self.calibrator = CalibratedClassifierCV(self.model, method='isotonic', cv='prefit')
        self.calibrator.fit(X_va, y_va)

        # Metrics
        p_tr = self.calibrator.predict_proba(X_tr)[:,1]
        p_va = self.calibrator.predict_proba(X_va)[:,1]
        self.train_brier = brier_score_loss(y_tr, p_tr)
        self.val_brier = brier_score_loss(y_va, p_va)
        self.train_logloss = log_loss(y_tr, p_tr, labels=[0,1])
        self.val_logloss = log_loss(y_va, p_va, labels=[0,1])
        self.is_trained = True
        return {
            'train_brier': self.train_brier,
            'val_brier': self.val_brier,
            'train_logloss': self.train_logloss,
            'val_logloss': self.val_logloss
        }

    def predict_proba(self, df):
        if not self.is_trained:
            raise RuntimeError('Model not trained')
        df2 = self.engineer_features(df)
        X = df2[self._feature_columns(df2)].fillna(df2[self._feature_columns(df2)].median())
        p = self.calibrator.predict_proba(X)[:,1]
        return p

    def select_bets(self, df, proba=None):
        df2 = df.copy()
        if proba is None:
            proba = self.predict_proba(df2)
        df2['pred_win_prob'] = proba
        df2['price'] = self._to_num(df2.get('average_price'))
        df2['ev'] = df2['pred_win_prob'] * (df2['price'] - 1.0) - (1.0 - df2['pred_win_prob'])

        def band(price):
            if pd.isna(price) or price <= 0:
                return 'na'
            if price < 3:
                return '<3'
            if price < 6:
                return '3-6'
            if price < 10:
                return '6-10'
            if price < 20:
                return '10-20'
            return '>20'

        df2['price_band'] = df2['price'].apply(band)
        df2['threshold'] = df2['price_band'].map(self.price_band_thresholds)
        df2['threshold'] = df2['threshold'].fillna(self.fallback_min_prob)
        df2['bet_flag'] = (df2['ev'] >= df2['threshold']).astype(int)
        return df2
