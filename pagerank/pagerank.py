import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    page_probability = {}
    k = (1 - damping_factor) / len(corpus.keys())
    for key in corpus.keys():
        page_probability[key] = k
    if corpus[page]:
        for key in corpus.keys():
            if key == page:
                for value in corpus[key]:
                    page_probability[value] += damping_factor / len(corpus[key])
    else:
        for key in corpus.keys():
            page_probability[key] += damping_factor / len(corpus.keys())
    return page_probability
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    ranks = {}
    samples = {}
    for key in corpus.keys():
        samples[key] = 0
    prev_sample = random.choice(list(corpus.keys()))[0]
    for i in range(n):
        prob = transition_model(corpus, prev_sample, damping_factor)
        next_sample = random.choices(list(prob.keys()), weights=tuple(prob.values()))[0]
        samples[prev_sample] += 1
        prev_sample = next_sample
    for key in samples.keys():
        ranks[key] = samples[key] / n
    return ranks
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    current_ranks = {}
    new_ranks = {}
    links = {}
    for key in corpus.keys():
        current_ranks[key] = 1 / len(corpus.keys())
        links[key] = [k for k, v in corpus.items() if key in v]
    for key in corpus.keys():
        if len(corpus[key]) == 0:
            for i in corpus.keys():
                links[i].append(key)
    new_ranks = calculation(corpus, damping_factor, current_ranks, links)
    return new_ranks
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # raise NotImplementedError


def calculation(corpus, damping_factor, probability, links):
    result = {}
    d = (1 - damping_factor) / len(corpus.keys())
    for key in corpus.keys():
        result[key] = d
    for key in corpus.keys():
        for page in links[key]:
            if len(corpus[page]) != 0:
                result[key] += damping_factor * (probability[page] / len(corpus[page]))
            else:
                result[key] += damping_factor * (probability[page] / len(corpus.keys()))
    for key in result.keys():
        if abs(result[key] - probability[key]) > 0.001:
            return calculation(corpus, damping_factor, result, links)
        else:
            return result


if __name__ == "__main__":
    main()
