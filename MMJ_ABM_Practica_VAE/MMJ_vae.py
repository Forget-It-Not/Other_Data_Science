# Implements auto-encoding variational Bayes.

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from data import load_mnist
from data import save_images as s_images
from autograd.misc import flatten # This is used to flatten the params (transforms a list into a numpy array)

# images is an array with one row per image, file_name is the png file on which to save the images

def save_images(images, file_name): return s_images(images, file_name, vmin = 0.0, vmax = 1.0)

# Sigmoid activiation function to estimate probabilities

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Relu activation function for non-linearity

def relu(x):    return np.maximum(0, x)

# This function intializes the parameters of a deep neural network

def init_net_params(layer_sizes, scale = 1e-2):

    """Build a (weights, biases) tuples for all layers."""

    return [(scale * npr.randn(m, n),   # weight matrix
             scale * npr.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

# This will be used to normalize the activations of the NN

# This computes the output of a deep neuralnetwork with params a list with pairs of weights and biases

def neural_net_predict(params, inputs):

    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""

    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)         # nonlinear transformation

    # Last layer is linear

    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb

    return outputs

# This implements the reparametrization trick

def sample_latent_variables_from_posterior(encoder_output):

    # Params of a diagonal Gaussian.

    D = np.shape(encoder_output)[-1] // 2
    mean, log_std = encoder_output[:, :D], encoder_output[:, D:]

    # Standard gaussian sample
    epsilon = npr.randn(*mean.shape)
    # Rescaling
    z = mean + epsilon*np.exp(log_std)

    return z

# This evlauates the log of the term that depends on the data

def bernoulli_log_prob(targets, logits):

    # logits are in R
    # Targets must be between 0 and 1

    # TODO compute the log probability of the targets given the generator output specified in logits
    # sum the probabilities across the dimensions of each image in the batch. The output of this function
    # should be a vector of size the batch size

    mat_probs = targets*sigmoid(logits) + (1-targets)*(1-sigmoid(logits))
    mat_log_probs = np.log(mat_probs)
    log_probs = np.sum(mat_log_probs, axis=1)

    return log_probs

# This evaluates the KL between q and the prior

def compute_KL(q_means_and_log_stds):

    D = np.shape(q_means_and_log_stds)[-1] // 2
    mean, log_std = q_means_and_log_stds[:, :D], q_means_and_log_stds[:, D:]

    # TODO compute the KL divergence between q(z|x) and the prior (use a standard Gaussian for the prior)
    # Use the fact that the KL divervence is the sum of KL divergence of the marginals if q and p factorize
    # The output of this function should be a vector of size the batch size

    KL = np.sum(0.5*(np.exp(log_std) + mean**2 - 1 - log_std), axis=1)

    return KL

# This evaluates the lower bound
def vae_lower_bound(gen_params, rec_params, data):

    # TODO compute a noisy estiamte of the lower bound by using a single Monte Carlo sample:

    # 1 - compute the encoder output using neural_net_predict given the data and rec_params
    # 2 - sample the latent variables associated to the batch in data
    #     (use sample_latent_variables_from_posterior and the encoder output)
    # 3 - use the sampled latent variables to reconstruct the image and to compute the log_prob of the actual data
    #     (use neural_net_predict for that)
    # 4 - compute the KL divergence between q(z|x) and the prior (use compute_KL for that)
    # 5 - return an average estimate (per batch point) of the lower bound by substracting the KL to the data dependent term

    encoder_output = neural_net_predict(rec_params, data)
    latent_variables = sample_latent_variables_from_posterior(encoder_output)
    generator_output = neural_net_predict(gen_params, latent_variables)
    data_log_prob = bernoulli_log_prob(data, generator_output)
    kl_divergence = compute_KL(encoder_output)
    average_estimate = np.mean(data_log_prob - kl_divergence)

    return average_estimate


if __name__ == '__main__':

    # Model hyper-parameters

    npr.seed(0) # We fix the random seed for reproducibility

    latent_dim = 50
    data_dim = 784  # How many pixels in each image (28x28).
    n_units = 200
    n_layers = 2


    gen_layer_sizes = [ latent_dim ] + [ n_units for i in range(n_layers) ] + [ data_dim ]
    rec_layer_sizes = [ data_dim ]  + [ n_units for i in range(n_layers) ] + [ latent_dim * 2 ]

    # Training parameters

    batch_size = 200
    num_epochs = 30
    learning_rate = 0.001

    print("Loading training data...")

    N, train_images, _, test_images, _ = load_mnist()

    # Parameters for the generator network p(x|z)

    init_gen_params = init_net_params(gen_layer_sizes)

    # Parameters for the recognition network p(z|x)

    init_rec_params = init_net_params(rec_layer_sizes)

    combined_params_init = (init_gen_params, init_rec_params)

    num_batches = int(np.ceil(len(train_images) / batch_size))

    # We flatten the parameters (transform the lists or tupples into numpy arrays)

    flattened_combined_params_init, unflat_params = flatten(combined_params_init)

    already_trained = True
    if not already_trained:

        # Actual objective to optimize that receives flattened params

        def objective(flattened_combined_params):

            combined_params = unflat_params(flattened_combined_params)
            data_idx = batch
            gen_params, rec_params = combined_params

            # We binarize the data

            on = train_images[ data_idx ,: ] > npr.uniform(size = train_images[ data_idx ,: ].shape)
            images = train_images[ data_idx, : ] * 0.0
            images[ on ] = 1.0

            return vae_lower_bound(gen_params, rec_params, images)

        # Get gradients of objective using autograd.

        objective_grad = grad(objective)
        flattened_current_params = flattened_combined_params_init

        # ADAM parameters

        # TODO write here the initial values for the ADAM parameters (including the m and v vectors)
        # you can use np.zeros_like(flattened_current_params) to initialize m and v

        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8

        t = 1
        m = np.zeros_like(flattened_current_params)
        v = np.zeros_like(flattened_current_params)

        # We do the actual training

        for epoch in range(num_epochs):

            elbo_est = 0.0

            for n_batch in range(int(np.ceil(N / batch_size))):

                batch = np.arange(batch_size * n_batch, np.minimum(N, batch_size * (n_batch + 1)))
                grad = objective_grad(flattened_current_params)

                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*grad**2
                mu = m/(1-beta1**t)
                vu = v/(1-beta2**t)
                flattened_current_params += learning_rate*mu/(np.sqrt(vu)+epsilon)

                # TODO Use the estimated noisy gradient in grad to update the paramters using the ADAM updates

                elbo_est += objective(flattened_current_params)

                t += 1

            print("Epoch: %d ELBO: %e" % (epoch, elbo_est / np.ceil(N / batch_size)))

        np.savetxt("trained_params.data", flattened_current_params)
    else:
        flattened_current_params = np.loadtxt("trained_params.data")

    # We obtain the final trained parameters
    gen_params, rec_params = unflat_params(flattened_current_params)

    # TODO Generate 25 images from prior (use neural_net_predict) and save them using save_images

    # Sample of latent variables -> sample from probability of positions
    latent_variables = npr.randn(25, latent_dim)
    generator_output = neural_net_predict(gen_params, latent_variables)
    generator_probs = sigmoid(generator_output)
    s_images(generator_probs, "images/temp")

    # TODO Generate image reconstructions for the first 10 test images (use neural_net_predict for each model)
    # and save them alongside with the original image using save_images

    # Reconstruction of images can be done simply with the final output of the
    # VAE and then choosing the value with highest prob (1 if sigmoid(logit)>0.5)
    first_10_test = test_images[:10]
    s_images(first_10_test, "images/test_real")
    encoder_output = neural_net_predict(rec_params, first_10_test)
    latent_variables = sample_latent_variables_from_posterior(encoder_output)
    generator_output = neural_net_predict(gen_params, latent_variables)
    generator_probs = sigmoid(generator_output)
    s_images(generator_probs, "images/test_pred")

    # TODO Generate 5 interpolations from the first test image to the second test image,
    # for the third to the fourth and so on until 5 interpolations
    # are computed in latent space and save them using save images.
    # Use a different file name to store the images of each iterpolation.
    # To interpolate from  image I to image G use a convex conbination. Namely,
    # I * s + (1-s) * G where s is a sequence of numbers from 0 to 1 obtained by numpy.linspace
    # Use mean of the recognition model as the latent representation.

    num_interpolations = 5
    for i in range(5):
        this_images = test_images[2*i:2*i+2]
        encoder_output = neural_net_predict(rec_params, this_images)
        this_latents = sample_latent_variables_from_posterior(encoder_output)

        reconstruction_list = []
        for s in np.linspace(0,1,20):
            mixed_latent = s*this_latents[0] + (1-s)*this_latents[1]
            reconstruction_list.append(mixed_latent)

        mixed_images = np.array(reconstruction_list)
        generator_output = neural_net_predict(gen_params, mixed_images)
        generator_probs = sigmoid(generator_output)
        s_images(generator_probs, f"images/interpolate_{i}")

        pass
