# hetengzhang
# csci 323
# winter 2022
# assgnment 3
import math
import string
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join
from lorem_text import lorem
from copy import deepcopy


def random_txt(m):
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(m))



def random_pattern(n, text):
    p = len(text)
    idx = random.randint(0, p - n)
    return text[idx:idx + n]


# https://www.geeksforgeeks.org/naive-algorithm-for-pattern-searching/
def brute_force(pat, text):
    M = len(pat)
    N = len(text)

    # A loop to slide pat[] one by one */
    for i in range(N - M + 1):
        j = 0

        # For current index i, check
        # for pattern match */
        while j < M:
            if text[i + j] != pat[j]:
                break
            j += 1

        if j == M:
            return


# https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/
def kmp_search(pat, text):
    M = len(pat)
    N = len(text)

    # create lps[] that will hold the longest prefix suffix
    # values for pattern
    lps = [0] * M
    j = 0  # index for pat[]

    # Preprocess the pattern (calculate lps[] array)
    compute_lps_array(pat, M, lps)

    i = 0  # index for txt[]
    while i < N:
        if pat[j] == text[i]:
            i += 1
            j += 1

        if j == M:

            j = lps[j - 1]

        # mismatch after j matches
        elif i < N and pat[j] != text[i]:
            # Do not match lps[0..lps[j-1]] characters,
            # they will match anyway
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1


def compute_lps_array(pat, M, lps):
    len = 0  # length of the previous longest prefix suffix

    lps[0]  # lps[0] is always 0
    i = 1

    # the loop calculates lps[i] for i = 1 to M-1
    while i < M:
        if pat[i] == pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            # This is tricky. Consider the example.
            # AAACAAAA and i = 7. The idea is similar
            # to search step.
            if len != 0:
                len = lps[len - 1]

                # Also, note that we do not increment i here
            else:
                lps[i] = 0
                i += 1


# https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching/
def rabin_karp(pat, text):
    d = 256
    q = 105
    M = len(pat)
    N = len(text)
    i = 0
    j = 0
    p = 0  # hash value for pattern
    t = 0  # hash value for txt
    h = 1

    # The value of h would be "pow(d, M-1)%q"
    for i in range(M - 1):
        h = (h * d) % q

    # Calculate the hash value of pattern and first window
    # of text
    for i in range(M):
        p = (d * p + ord(pat[i])) % q
        t = (d * t + ord(text[i])) % q

    # Slide the pattern over text one by one
    for i in range(N - M + 1):
        # Check the hash values of current window of text and
        # pattern if the hash values match then only check
        # for characters on by one
        if p == t:
            # Check for characters one by one
            for j in range(M):
                if text[i + j] != pat[j]:
                    break
                else:
                    j += 1

            # if p == t and pat[0...M-1] = txt[i, i+1, ...i+M-1]
            if j == M:
                return 0

        # Calculate hash value for next window of text: Remove
        # leading digit, add trailing digit
        if i < N - M:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + M])) % q

            # We might get negative values of t, converting it to
            # positive
            if t < 0:
                t = t + q


def native_search(pat, txt):
    size = len(pat)
    for i in range(len(txt) - size + 1):
        if txt[i: i + size] == pat:
           return pat,txt


def plot_times_line_graph(dict_searches):
    for search in dict_searches:
        x = dict_searches[search].keys()
        y = dict_searches[search].values()
        plt.plot(x, y, label=search)
    plt.legend()
    plt.title("Run Time of Search Algorithms")
    plt.xlabel("Number of Elements")
    plt.ylabel("Time for 100 Trials")
    plt.savefig("search_graph.png")
    plt.show()


def plot_times_bar_graph(dict_searches, sizes, searches):
    search_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])

    for search in searches:
        search_num += 1
        d = dict_searches[search.__name__]
        x_axis = [j + 0.05 * search_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=0.07, alpha=1, label=search.__name__)
    plt.legend()
    plt.title("Run Time of Search Algorithms")
    plt.xlabel("Number of Elements")
    plt.ylabel("Time for 100 Trials")
    plt.savefig("search_graph.png")
    plt.show()


def main():
    m = 15000
    trials = 50
    dict_searches = {}
    searches = [native_search, brute_force, kmp_search, rabin_karp]
    for search in searches:
        dict_searches[search.__name__] = {}
    sizes = [1000, 2000, 3000, 4000, 5000]
    for size in sizes:
        for search in searches:
            dict_searches[search.__name__][size] = 0
        for trials in range(1, trials + 1):
            text = random_txt(m)
            pat = random_pattern(size, text)
            for search in searches:
                start_time = time.time()
                search(pat, text)
                end_time = time.time()
                net_time = end_time - start_time
                dict_searches[search.__name__][size] += 1000 * net_time

                if trials == trials-1:
                    dict_searches[search.__name__][size] /= trials
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.DataFrame.from_dict(dict_searches).T
    print(df)
    print("average time", net_time / trials)
    # plot_times_line_graph(dict_searches)
    plot_times_bar_graph(dict_searches, sizes, searches)


if __name__ == '__main__':
    main()
