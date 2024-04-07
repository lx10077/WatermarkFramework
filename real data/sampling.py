import torch
from IPython import embed
from alternative_prf_schemes import prf_lookup

def seed_rng(generator, tokens, seeding_scheme="minhash_prf", hash_key=15485863, c=5):
    # Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched.
    # Borrowed from 
    # https://github.com/jwkirchenbauer/lm-watermarking/blob/main/watermark_reliability_release/watermark_processor.py
    # tokens should be in the shape of (1, current_length)

    assert tokens.shape[-1] >= c, f"seeding_scheme={seeding_scheme} requires at least a {c} token prefix sequence to seed rng"
    prf_key = prf_lookup[seeding_scheme](tokens[0][-c:], salt_key=hash_key)
    generator.manual_seed(prf_key)


## For Gumbel-max watermarks
def gumbel_key_func(generator,inputs,vocab_size, key, c, seeding_scheme):
    # add randonseed
    xis = []
    pis = []
    for k in range(inputs.shape[0]):
        seed_rng(generator, inputs[k].unsqueeze(0), seeding_scheme=seeding_scheme, hash_key=key, c=c) # This function require inputs of the shape (1, Length)
        xi = torch.rand(size=(1,vocab_size), generator=generator)
        pi = torch.arange(vocab_size)
        xis.append(xi)
        pis.append(pi)
    xis=torch.vstack(xis)
    pis=torch.vstack(pis)
    return xis,pis

def gumbel_sampling(probs,pi,xi):
    return torch.argmax(xi ** (1/torch.gather(probs, 1, pi)),axis=1).unsqueeze(-1)

def gumbel_Y(s, pi, xi):
    xi_samp = torch.gather(xi,-1,s.cpu()).squeeze()
    return xi_samp


## For inverse transform watermarks
def transform_key_func(generator,inputs,vocab_size, key, c, seeding_scheme):
    batch_size = inputs.shape[0] # batch_size must be 1
    assert batch_size == 1, "Batch size should be 1 for the inverse transform watermark!"
    # add randonseed
    xis = []
    pis = []
    for _ in range(batch_size):
        seed_rng(generator, inputs, seeding_scheme=seeding_scheme, hash_key=key, c=c)
        xi = torch.rand(size=(batch_size,1), generator=generator)
        pi = torch.randperm(vocab_size, generator=generator)
        xis.append(xi)
        pis.append(pi)
    xis=torch.vstack(xis)
    pis=torch.vstack(pis)
    return xis,pis

def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


def transform_sampling(probs,pi,xi):
    inv_pi = inverse_permutation(pi.squeeze()).unsqueeze(0)
    cdf = torch.cumsum(torch.gather(probs, 1, inv_pi), 1)
    s = torch.gather(inv_pi, 1, torch.searchsorted(cdf, xi))
    return s


def transform_Y(s, pi, xi):
    ## For dif: Y = -|U - eta|. 
    ## Unlike the form we introduced in our paper, we add a minus in experiments
    ## In this way, we will have E_0 Y < E_1 Y.
    vocab_size = pi.shape[1]
    s_samp = torch.gather(pi,-1,s.cpu()).squeeze() 
    return -torch.abs(xi-(s_samp-1)/(vocab_size-1)), xi, (s_samp-1)/(vocab_size-1)
