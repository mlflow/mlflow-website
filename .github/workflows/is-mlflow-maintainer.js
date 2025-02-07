const MLFLOW_CORE_MAINTAINERS = [
  "B-Step62",
  "BenWilson2",
  "daniellok-db",
  "harupy",
  "mlflow-automation",
  "serena-ruan",
  "WeichenXu123",
  "TomeHirata",
]

const isMLflowMaintainer = (username) => {
    return MLFLOW_CORE_MAINTAINERS.includes(username);
}

module.exports = {
    isMLflowMaintainer,
}
