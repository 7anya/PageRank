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
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # probability-list to initialize weights for random no. generator
    p = [damping_factor, 1 - damping_factor]
    model = dict()
    linked = set()
    unlinked = set()

    for key, value in corpus.items():

        model[key] = 0.0

        if key != page:
            continue

        linked = value
        unlinked = set()
        for key in corpus:
            if key in linked:
                continue
            unlinked.add(key)

    linked_count = len(linked)
    unlinked_count = len(unlinked)
    total = linked_count + unlinked_count
    base_prob = p[1] / total
    for key in model:

        if key in linked:
            model[key] = base_prob + damping_factor / linked_count
        else:
            model[key] = base_prob

    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    rank = dict()

    for key in corpus:
        rank[key] = 0.0

    sample = random.choices(list(corpus))[0]
    rank[sample] += (1 / n)

    for i in range(1, n):

        model = transition_model(corpus, sample, damping_factor)
        next_pages = []
        probabilities = []
        for key, value in model.items():
            next_pages.append(key)
            probabilities.append(value)

        sample = random.choices(next_pages, weights=probabilities)[0]
        rank[sample] += (1 / n)

    return rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    ranks = dict()

    threshold = 0.0005

    N = len(corpus)
    for key in corpus:
        ranks[key] = 1 / N
    while True:

        count = 0

        for key in corpus:

            new = (1 - damping_factor) / N
            sigma = 0

            for page in corpus:

                if key in corpus[page]:
                    num_links = len(corpus[page])
                    sigma = sigma + ranks[page] / num_links

            sigma = damping_factor * sigma
            new += sigma

            if abs(ranks[key] - new) < threshold:
                count += 1

            ranks[key] = new

        if count == N:
            break

    return ranks


if __name__ == "__main__":
    main()
