from sklearn.preprocessing import StandardScaler

def apply_standard_scaler(gradients):
    scaler = StandardScaler()
    Y = scaler.fit_transform(gradients)
    return Y, scaler
