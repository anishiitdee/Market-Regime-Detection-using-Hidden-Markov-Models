import numpy as np

def norm_pdf(x, m, v):
    x, m, v = np.longdouble(x), np.longdouble(m), np.longdouble(v)
    return (np.exp((-(x - m)**2) / (v * 2)) / np.sqrt(2 * np.pi * v))


def algo1(A, pi, Y, mu, sig):
    n_s, n_o = len(A), len(Y)
    alphas = np.zeros((n_o, n_s), dtype=np.longdouble)

    i = 0
    while i < n_s:
        alphas[0][i] = (pi[i] * norm_pdf(Y[0], mu[i], sig[i]))
        i += 1

    t = 1
    while t < n_o:
        i = 0
        while i < n_s:
            alphas[t][i] = np.sum(norm_pdf(Y[t], mu[i], sig[i]) * A[:, i] * alphas[t - 1])
            i += 1
        t += 1

    return alphas

def algo2(A, pi, Y, mu, sig):
    n_s, n_o = len(A), len(Y)
    betas = np.zeros((n_o, n_s), dtype=np.longdouble)

    i = 0
    while i < n_s:
        betas[n_o - 1][i] = 1
        i += 1
    
    t = n_o - 2
    while t >= 0:
        i = 0
        while i < n_s:
            betas[t][i] = np.sum(betas[t + 1] * A[i] * norm_pdf(Y[t + 1], mu, sig))
            i += 1
        t -= 1

    return betas

def baum_welch(A, pi, Y, mu, sig):
    max_iter, tol = 10000, np.longdouble(1e-8)

    iterations = 0
    while iterations < max_iter:
        alphas = algo1(A, pi, Y, mu, sig)
        betas = algo2(A, pi, Y, mu, sig)
        
        n_s, n_o = len(A), len(Y)
        zetas = np.zeros((n_o - 1, n_s, n_s), dtype=np.longdouble)
        
        t = 0
        while t < n_o - 1:
            zetas[t] = alphas[t, :, None] * A * betas[t + 1, None, :] * norm_pdf(Y[t + 1], mu[None, :], sig[None, :])
            zetas[t] /= np.sum(zetas[t])
            t += 1
        
        gammas = np.zeros((n_o, n_s), dtype=np.longdouble)
        t = 0
        while t < n_o:
            gammas[t] = alphas[t] * betas[t]
            gammas[t] /= np.sum(gammas[t])
            t += 1

        new_pi = gammas[0]
        new_A = ((np.sum(zetas, axis=0)) / (np.sum(gammas[:-1], axis=0)[:, None]))
        new_mu = ((np.dot(Y, gammas)) / (np.sum(gammas, axis=0)))
        new_sig = np.array([np.dot((Y - new_mu[i])**2, gammas[:, i]) / np.sum(gammas[:, i]) for i in range(2)])
        
        if np.allclose(pi, new_pi, atol=tol) and np.allclose(A, new_A, atol=tol) and np.allclose(mu, new_mu, atol=tol) and np.allclose(sig, new_sig, atol=tol):
            break

        A, pi, mu, sig = new_A, new_pi, new_mu, new_sig

        iterations += 1

    return gammas, new_mu

def main():
    A = np.array(list(map(np.longdouble, input().split()))).reshape(2, 2)
    pi = np.array(list(map(np.longdouble, input().split())))
    n = int(input())
    Y = np.array([np.longdouble(input()) for _ in range(n)])
    mu = np.array([0, 0], dtype=np.longdouble)
    sig = np.array([1, 1], dtype=np.longdouble)

    gamma, mu = baum_welch(A, pi, Y, mu, sig)

    bull_state = 1 if mu[0] <= mu[1] else 0    
    for prob in gamma:
        if (prob[bull_state] > prob[1 - bull_state]):
            print("Bull")
        else:
            print("Bear")

if __name__ == "__main__":
    main()
