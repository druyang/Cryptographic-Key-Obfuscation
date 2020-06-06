import random
import sys
import struct
from gmpy2 import invert
import math
import numpy as np

def gcd(a, b):
    '''
    Euclid's algorithm for determining the greatest common divisor
    Use iteration to make it faster for larger integers
    '''
    while b != 0:
        a, b = b, a % b
    return a


def modinv(a, m):

    return int(invert(a, m))

    # g, x, y = egcd(a, m)
    # if g != 1:
    #     raise Exception('modular inverse does not exist')
    # else:
    #     return x % m


def rabinMiller(num):
    '''
    Tests to see if a number is prime.
    Returns True or False
    '''

    if num % 2 == 0:
        return False

    s = num - 1
    t = 0
    while s % 2 == 0:
        # keep halving s while it is even (and use t
        # to count how many times we halve s)
        s = s // 2
        t += 1

    for trials in range(5):  # try to falsify num's primality 5 times
        a = random.randrange(2, num - 1)
        v = pow(a, s, num)

        if v != 1:  # this test does not apply if v is 1.
            i = 0
            while v != (num - 1):
                if i == t - 1:
                    return False
                else:
                    i = i + 1
                    v = (v ** 2) % num
    return True


def generate_keypair(keysize):

    # Size of N must be size(P) + size(Q) + 1

    upper_bound = 2**(keysize/2 - 1) - 1
    lower_bound = 2**(keysize/2 - 2) - 1

    # Generate Prime Values for p and q
    while(True):
        p = random.randint(lower_bound, upper_bound)

        if rabinMiller(p):
            break

    while(True):
        q = random.randint(lower_bound, upper_bound)

        if rabinMiller(q) and q != p:
            break

    # n = pq
    n = p * q

    # Phi is the totient of n
    phi = (p-1) * (q-1)

    # Choose an integer e such that e and phi(n) are coprime
    e = random.randrange(1, phi)

    # Use Euclid's Algorithm to verify that e and phi(n) are comprime
    g = gcd(e, phi)
    while g != 1:
        e = random.randrange(1, phi)
        g = gcd(e, phi)

    # Use Extended Euclid's Algorithm to generate the private key
    d = modinv(e, phi)

    # Return public and private keypair
    # Public key is (e, n) and private key is (d, n)
    return ((e, n), (d, n))


def encrypt(pk, plaintext):
    # Unpack the key into it's components
    key, n = pk
    # Convert each letter in the plaintext to numbers based on the character
    # using a^b mod m
    cipher = [(ord(char) ** key) % n for char in plaintext]
    # Return the array of bytes
    return cipher


def decrypt(pk, ciphertext):
    # Unpack the key into its components
    key, n = pk
    # Generate the plaintext based on the ciphertext and key using a^b mod m
    plain = [chr((char ** key) % n) for char in ciphertext]
    # Return the array of bytes as a string
    return ''.join(plain)


def gen_matrix(pk, sidelen=16):
    """
    Compute two sidelen x sidelen matricies which can be multiplied in order to compute the private key.
    After multiplying these matricies, the resultant matrix is reduced to a single value, and an adjustment
    , or intercept value is added to the reduced sum to compute the private key.
    """
    
    mean = int(math.sqrt(pk / (sidelen**4))) # mean of the standard random normal distribution
    sd = mean // 3  # standard deviation of the random normal distribution
    
    # generates a sidelen x sidelen random normal matrix with values distributed around base_value
    matrix1 = np.random.normal(mean, sd, (sidelen, sidelen)).astype(int)
    matrix2 = np.random.normal(mean, sd, (sidelen, sidelen)).astype(int)    
    result = (matrix1 * matrix2).sum()
    adjustment = pk - result
    print("result = {}, adjustment = {}, so PK = {}".format(result, adjustment, result + adjustment))
    
    return (matrix1, matrix2, adjustment)


def write_key_computation_info(pk, fo, sidelen=16):

    matrix1, matrix2, adjustment = gen_matrix(pk, sidelen)
    print('Matrix sidelen:', sidelen, '\n', file=fo)
    print('adjustment:',adjustment, '\n', file=fo)
    print('Matrix 1:', matrix1.flatten().tolist(), '\n', file=fo)
    print('Matrix 2:', matrix2.flatten().tolist(), '\n', file=fo)


if __name__ == '__main__':
    '''
    Detect if the script is being run directly by the user
    '''
    keysize = int(sys.argv[1])
    (e, n), (d, n) = generate_keypair(keysize)
    fo = open('key.txt', 'w') 
    print("const unsigned ll e = "+ str(e) + ";\n", file=fo)
    print("const unsigned ll d = "+ str(d) + ";\n", file=fo)
    print("const unsigned ll n = "+ str(n) + ";\n", file=fo)
    write_key_computation_info(d, fo)
    fo.close()
