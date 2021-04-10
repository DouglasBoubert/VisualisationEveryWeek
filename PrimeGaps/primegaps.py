#https://codegolf.stackexchange.com/questions/10701/fastest-code-to-find-the-next-prime

import sys
import numpy as np
from numba import njit
from numba.typed import List
import tqdm

limit_order = int(sys.argv[1])
step_order = int(sys.argv[2])
max_gap = int(sys.argv[3])
N_core = int(sys.argv[4])
N_step = pow(10,limit_order-step_order)
step_size = pow(10,step_order)


# legendre symbol (a|m)
# note: returns m-1 if a is a non-residue, instead of -1
@njit
def legendre(a, m):
    return pow(a, (m-1) >> 1) % m

# strong probable prime
@njit
def is_sprp(n, b=2):
    d = n-1
    s = 0
    while d&1 == 0:
        s += 1
        d >>= 1

    #x = pow(b, d, n)
    x = pow(b, d) % n
    if x == 1 or x == n-1:
        return True

    for r in range(1, s):
        x = (x * x)%n
        if x == 1:
            return False
        elif x == n-1:
            return True

    return False

# lucas probable prime
# assumes D = 1 (mod 4), (D|n) = -1
@njit
def is_lucas_prp(n, D):
    P = 1
    Q = (1-D) >> 2

    # n+1 = 2**r*s where s is odd
    s = n+1
    r = 0
    while s&1 == 0:
        r += 1
        s >>= 1

    # calculate the bit reversal of (odd) s
    # e.g. 19 (10011) <=> 25 (11001)
    t = 0
    while s > 0:
        if s&1:
            t += 1
            s -= 1
        else:
            t <<= 1
            s >>= 1

    # use the same bit reversal process to calculate the sth Lucas number
    # keep track of q = Q**n as we go
    U = 0
    V = 2
    q = 1
    # mod_inv(2, n)
    inv_2 = (n+1) >> 1
    while t > 0:
        if t&1 == 1:
            # U, V of n+1
            U, V = ((U + V) * inv_2)%n, ((D*U + V) * inv_2)%n
            q = (q * Q)%n
            t -= 1
        else:
            # U, V of n*2
            U, V = (U * V)%n, (V * V - 2 * q)%n
            q = (q * q)%n
            t >>= 1

    # double s until we have the 2**r*sth Lucas number
    while r > 0:
            U, V = (U * V)%n, (V * V - 2 * q)%n
            q = (q * q)%n
            r -= 1

    # primality check
    # if n is prime, n divides the n+1st Lucas number, given the assumptions
    return U == 0

# primes less than 212
small_primes = List([
        2,    3,    5,    7, 11, 13, 17, 19, 23, 29,
     31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
     73, 79, 83, 89, 97,101,103,107,109,113,
    127,131,137,139,149,151,157,163,167,173,
    179,181,191,193,197,199,211])

# pre-calced sieve of eratosthenes for n = 2, 3, 5, 7
indices = np.array([
        1, 11, 13, 17, 19, 23, 29, 31, 37, 41,
     43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
     89, 97,101,103,107,109,113,121,127,131,
    137,139,143,149,151,157,163,167,169,173,
    179,181,187,191,193,197,199,209],dtype=int)

# distances between sieve values
offsets = np.array([
    10, 2, 4, 2, 4, 6, 2, 6, 4, 2, 4, 6,
     6, 2, 6, 4, 2, 6, 4, 6, 8, 4, 2, 4,
     2, 4, 8, 6, 4, 6, 2, 4, 6, 2, 6, 6,
     4, 2, 4, 6, 2, 6, 4, 2, 4, 2,10, 2],dtype=int)

max_int = 2147483647

# an 'almost certain' primality check
@njit
def is_prime(n,small_primes):
    if n < 212:
        return n in small_primes

    for p in small_primes:
        if n%p == 0:
            return False

    # if n is a 32-bit integer, perform full trial division
    if n <= max_int:
        i = 211
        while i*i < n:
            for o in offsets:
                i += o
                if n%i == 0:
                    return False
        return True

    # Baillie-PSW
    # this is technically a probabalistic test, but there are no known pseudoprimes
    if not is_sprp(n): return False
    a = 5
    s = 2
    while legendre(a, n) != n-1:
        s = -s
        a = s-a
    return is_lucas_prp(n, a)

# next prime strictly larger than n
@njit
def next_prime(n,small_primes):
    if n < 2:
        return 2
    # first odd larger than n
    n = (n + 1) | 1
    if n < 212:
        while True:
            if n in small_primes:
                return n
            n += 2

    # find our position in the sieve rotation via binary search
    x = int(n%210)
    s = 0
    e = 47
    m = 24
    while m != e:
        if indices[m] < x:
            s = m
            m = (s + e + 1) >> 1
        else:
            e = m
            m = (s + e) >> 1

    i = int(n + (indices[m] - x))
    # adjust offsets
    #offs = offsets[m:]+offsets[:m]
    offs = np.roll(offsets,-m)
    while True:
        for o in offs:
            if is_prime(i,small_primes):
                return i
            i += o
            
@njit
def prime_freq(start,end,small_primes,store):
    n = max(2,start-max_gap)
    
    while n < start:
        n = next_prime(n,small_primes)
    
    while n < end:
        nextn = next_prime(n,small_primes)
        diff = nextn - n
        n = nextn
        store[diff] += 1

# Run grid
import multiprocessing

def mp_worker(args):
    step_index = args[0]
    
    start = step_index*step_size
    end = start+step_size
    store = np.zeros(max_gap+1,dtype=int)
    prime_freq(start,end,small_primes,store)
    
    return (store,)

def mp_handler():
    pool = multiprocessing.Pool(N_core)
    _input = [(step_index,) for step_index in range(N_step)]
    result = list(tqdm.tqdm(pool.imap(mp_worker, _input), total=N_step))
    return result

output = mp_handler()
gap_distribution = np.stack([output[step_index][0] for step_index in range(N_step)])

##### Output
import h5py
with h5py.File(f"primegaps_{limit_order}.hdf5", "w") as f:
    f.create_dataset("counts", data=gap_distribution, compression="gzip", compression_opts=9, chunks = True, dtype = np.uint64, fletcher32 = False, shuffle = True, scaleoffset=0)
    f.create_dataset("starts", data=np.array([step_index*step_size for step_index in range(N_step)]))
    f.create_dataset("ends", data=np.array([step_index*step_size + step_size for step_index in range(N_step)]))
    f.create_dataset("gaps", data=np.arange(max_gap+1))
#output_box = {'log_integrals':_log_integrals,'n':n,'k':k,'alpha':np.logspace(-1,4,100),'beta':np.logspace(-1,4,100)}
#save_as_pickled_object(output_box,'censored_binomial_fully_log_grid_100.p')