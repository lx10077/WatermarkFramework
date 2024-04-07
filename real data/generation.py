import torch


def generate_inv(model,prompts,vocab_size,m,key_func,sampler, Y_func, key=23333,c=5, seeding_scheme="minhash_prf",temperature=0.1):
    generator = torch.Generator()
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None

    Ys = []
    Us = []
    etas = []
    top_probs = []
    for _ in range(m): 
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1]/temperature, dim=-1).cpu()
        top_prob = torch.max(probs, axis=1)[0].unsqueeze(0)
        xi, pi = key_func(generator,inputs, vocab_size, key, c, seeding_scheme)  
        tokens = sampler(probs, pi, xi).to(model.device) 
        Y, U, eta = Y_func(tokens, pi, xi)

        inputs = torch.cat([inputs, tokens], dim=-1)

        Ys.append(Y.unsqueeze(0))
        Us.append(U.unsqueeze(0))
        etas.append(eta.unsqueeze(0))
        top_probs.append(top_prob)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    Ys = torch.vstack(Ys)
    Us = torch.vstack(Us)
    etas = torch.vstack(etas)
    top_probs = torch.vstack(top_probs)
    return inputs.detach().cpu(), Ys.detach().cpu(),Us.detach().cpu(), etas.detach().cpu(),top_probs.detach().cpu()


def generate_gum(model,prompts,vocab_size,m,key_func,sampler, Y_func, key=23333,c=5, seeding_scheme="minhash_prf",temperature=0.1):
    generator = torch.Generator()
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None

    Ys = []
    top_probs = []
    for _ in range(m): # 
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1]/temperature, dim=-1).cpu()
        top_prob = torch.max(probs, axis=1)[0].unsqueeze(0)
        xi, pi = key_func(generator,inputs, vocab_size, key, c, seeding_scheme)  
        tokens = sampler(probs, pi, xi).to(model.device) 
        Y = Y_func(tokens, pi, xi)

        inputs = torch.cat([inputs, tokens], dim=-1)

        Ys.append(Y.unsqueeze(0))
        top_probs.append(top_prob)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    Ys = torch.vstack(Ys)
    top_probs = torch.vstack(top_probs)
    return inputs.detach().cpu(), Ys.detach().cpu(), top_probs.detach().cpu(), 


# generate unwatermarked completions of token length m given list of prompts
def generate_rnd(prompts,m,model):
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1], dim=-1)
        
        tokens = torch.multinomial(probs,1)
        inputs = torch.cat([inputs, tokens], dim=1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
    
    return inputs.detach().cpu()
