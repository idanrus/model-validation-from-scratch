import warnings
from itertools import product

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------- Разбиение на train / valid / test ----------

def data_split(X_, y_, test_size, shuffle=True, random_state=None):
    """Случайное разбиение на train и test. С фиксированным random_state результат всегда один и тот же."""
    X = X_.copy()
    y = y_.copy()
    if shuffle:
        indexes = X.sample(frac=1, random_state=random_state).index
    else:
        indexes = X.index
    n = X.shape[0]
    split = int(n * (1 - test_size))
    return X.loc[indexes[:split]], X.loc[indexes[split:]], y.loc[indexes[:split]], y.loc[indexes[split:]]


def data_split_valid(X_, y_, test_size, validation_size, shuffle=True, random_state=None):
    """Случайное разбиение на train, valid и test. Размеры задаются долями от всего датасета."""
    X_train, X_test, y_train, y_test = data_split(X_, y_, test_size=test_size,
                                                  shuffle=shuffle, random_state=random_state)
    X_train, X_valid, y_train, y_valid = data_split(X_train, y_train, test_size=validation_size / (1 - test_size),
                                                    shuffle=shuffle, random_state=random_state)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def date_split(X, y, date_split, date_column=None):
    """Out-of-time: всё до date_split включительно идёт в train, остальное в test."""
    if date_column is None:
        mask = X.index <= date_split
    else:
        mask = X[date_column] <= date_split
    return X[mask], X[~mask], y[mask], y[~mask]


def date_split_valid(X, y, validation_date, test_date, date_column=None):
    """Out-of-time на три части: train <= validation_date < valid < test_date <= test."""
    if validation_date >= test_date:
        raise ValueError("validation_date должна быть меньше test_date")
    if date_column is None:
        dates = X.index
    else:
        dates = X[date_column]
    mask_train = dates <= validation_date
    mask_valid = (dates > validation_date) & (dates < test_date)
    mask_test = dates >= test_date
    return X[mask_train], X[mask_valid], X[mask_test], y[mask_train], y[mask_valid], y[mask_test]


# ---------- Кросс-валидация ----------
# Все классы повторяют интерфейс sklearn: split(X, y, groups) возвращает позиционные индексы.

class KFold_custom:
    def __init__(self, n_splits, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        n_splits = self.n_splits
        if n_splits > n_samples:
            raise ValueError('n_splits должно быть не больше n_samples')
        indexes = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            indexes = rng.permutation(indexes)
        # первые n_samples % n_splits фолдов на один объект больше, как в sklearn
        fold_sizes = np.full(n_splits, n_samples // n_splits)
        fold_sizes[:n_samples % n_splits] += 1
        curr_ind = 0
        for size in fold_sizes:
            # берём позиции curr_ind:curr_ind+size в (возможно перемешанном) массиве индексов
            mask = np.zeros(n_samples, dtype=bool)
            mask[indexes[curr_ind:curr_ind + size]] = True
            test_ind = np.arange(n_samples)[mask]
            train_ind = np.arange(n_samples)[~mask]
            curr_ind += size
            yield train_ind, test_ind


class GroupKFold_custom:
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_splits = self.n_splits
        groups = np.asarray(groups)
        a, b = np.unique(groups, return_counts=True)
        if len(a) < n_splits:
            raise ValueError('количество групп не может быть меньше количества сплитов')
        if len(X) != len(groups):
            raise ValueError('несоответсвие длин')
        # жадно: самую большую из оставшихся групп кладём в самый маленький фолд
        order = np.argsort(-b, kind='stable')
        a = a[order]
        b = b[order]
        grouping = [[] for _ in range(n_splits)]
        sizes = np.zeros(n_splits, dtype=int)
        for value, count in zip(a, b):
            min_size_group = np.argmin(sizes)
            grouping[min_size_group].append(value)
            sizes[min_size_group] += count
        for group in grouping:
            mask = np.isin(groups, group)
            train_idx = np.where(~mask)[0]
            test_idx = np.where(mask)[0]
            yield train_idx, test_idx


class StratifiedKFold_custom:
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self):
        return self.n_splits

    def split(self, X, y, groups=None):
        y = np.asarray(y)
        unique, counts = np.unique(y, return_counts=True)
        if counts.min() < self.n_splits:
            warnings.warn(
                f"The least populated class in y has only {counts.min()} "
                f"members, which is less than n_splits={self.n_splits}.",
                UserWarning
            )
        # индексы каждого класса делим на n_splits частей, i-й фолд собираем из i-х частей всех классов
        splits = [np.array_split(np.where(y == u)[0], self.n_splits) for u in unique]
        n_samples = len(y)
        all_indices = np.arange(n_samples)
        for i in range(self.n_splits):
            test_idx = np.sort(np.concatenate([s[i] for s in splits]))
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            yield all_indices[train_mask], test_idx


class TimeSeriesSplit_custom:
    def __init__(self, n_splits, date_field=None):
        self.n_splits = n_splits
        self.date_field = date_field

    def get_n_splits(self):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        if self.date_field is not None:
            # позиции строк в порядке возрастания даты: индексы относятся к исходному X, сортировать его заранее не нужно
            indices = np.argsort(X[self.date_field].values, kind='stable')
        else:
            indices = np.arange(n_samples)
        test_size = n_samples // (self.n_splits + 1)
        # остаток от деления уходит в первый train, как в sklearn
        first_test = n_samples - self.n_splits * test_size
        for test_start in range(first_test, n_samples, test_size):
            yield indices[:test_start], indices[test_start:test_start + test_size]


# ---------- Отбор признаков ----------

def simple_feature_selection(X_, y_, nan_threshhold=0.5, cnt_features=10):
    """Убираем признаки с долей пропусков выше порога, остальные сортируем по |корреляции| с таргетом."""
    nan_ratio = X_.isna().mean()
    features_after_nan = nan_ratio[nan_ratio <= nan_threshhold].index
    X = X_[features_after_nan]
    features = X.corrwith(y_).abs().sort_values(ascending=False).iloc[:cnt_features].index.tolist()
    return features


def permutation_importance(model, X, y, metric=r2_score, n_features=10, random_state=42):
    """Важность признака = насколько падает метрика, если перемешать его значения."""
    base_score = metric(y, model.predict(X))
    importances = {}
    rng = np.random.RandomState(random_state)
    for col in X.columns:
        X_permuted = X.copy()
        X_permuted[col] = rng.permutation(X_permuted[col].values)
        shuffled_score = metric(y, model.predict(X_permuted))
        importances[col] = base_score - shuffled_score
    importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top_features = [x[0] for x in importances[:n_features]]
    return top_features, importances


# ---------- Подбор гиперпараметров ----------

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


scoring_dict = {
    'r2': r2_score,
    'MAE': mean_absolute_error,
    'RMSE': rmse
}


class custom_GridSearchCV:
    """Перебор всех комбинаций параметров, каждую оцениваем на кросс-валидации."""

    def __init__(self, model, params, scoring='r2', cv=None):
        self.model = model
        self.params = params
        self.scoring = scoring
        self.metric = scoring_dict[scoring]
        self.cv = cv if cv is not None else KFold_custom(n_splits=5)

    def get_param_list(self):
        keys = self.params.keys()
        return [dict(zip(keys, values)) for values in product(*self.params.values())]

    def cv_score(self, par, X_, y_):
        curr_metric = 0
        for train_index, test_index in self.cv.split(X_, y_):
            model = self.model.__class__(**par)
            model.fit(X_.iloc[train_index], y_.iloc[train_index])
            pred = model.predict(X_.iloc[test_index])
            curr_metric += self.metric(y_.iloc[test_index], pred)
        return curr_metric / self.cv.get_n_splits()

    def fit(self, X_, y_):
        # r2 максимизируем, ошибки минимизируем
        greater_is_better = self.scoring == 'r2'
        self.best_score_ = -np.inf if greater_is_better else np.inf
        self.best_params_ = {}
        for param in self.get_param_list():
            par = {**self.model.get_params(), **param}
            curr_metric = self.cv_score(par, X_, y_)
            if (curr_metric > self.best_score_) if greater_is_better else (curr_metric < self.best_score_):
                self.best_params_ = par
                self.best_score_ = curr_metric
        self.best_estimator_ = self.model.__class__(**self.best_params_)
        self.best_estimator_.fit(X_, y_)
        return self

    def predict(self, X_):
        return self.best_estimator_.predict(X_)


class custom_RandomGridSearchCV(custom_GridSearchCV):
    """То же самое, но проверяем только n_iter случайных комбинаций."""

    def __init__(self, model, params, n_iter=20, scoring='r2', cv=None, random_state=None):
        super().__init__(model, params, scoring=scoring, cv=cv)
        self.n_iter = n_iter
        self.random_state = random_state

    def get_param_list(self):
        rng = np.random.RandomState(self.random_state)
        return [{k: rng.choice(v) for k, v in self.params.items()} for _ in range(self.n_iter)]


# ---------- Метрики ----------

def get_metrics(name, y_true_list, pred_list, parts=('train', 'valid', 'test')):
    """Строка таблицы результатов: MAE, RMSE и R2 на каждой части выборки."""
    row = {'model': name}
    for part, y_true, pred in zip(parts, y_true_list, pred_list):
        row[f'MAE {part}'] = mean_absolute_error(y_true, pred)
        row[f'RMSE {part}'] = rmse(y_true, pred)
        row[f'R2 {part}'] = r2_score(y_true, pred)
    return row
