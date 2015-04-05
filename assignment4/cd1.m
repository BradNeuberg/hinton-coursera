function ret = cd1(rbm_w, visible_data)
  % <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
  % <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
  % The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

  % Question 8: Treat real-valued image data as conditional probabilities for a visible state.
  visible_data = sample_bernoulli(visible_data);

  % Given some visible data, clamp it and produce the probabilities of the hidden units.
  hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_data);

  % Take these hidden state probabilities and produce an actual sample state for the hidden units.
  hidden_sample = sample_bernoulli(hidden_probabilities);

  % Using the visible state data and the sampled hidden state, produce the configuration
  % goodness gradient.
  data_goodness = configuration_goodness_gradient(visible_data, hidden_sample);

  % Now reconstruct the visible state probabilities using the actual hidden sample states.
  visible_probabilities = hidden_state_to_visible_probabilities(rbm_w, hidden_sample);

  % Turn these visible probabilities into actual visible state samples.
  visible_sample = sample_bernoulli(visible_probabilities);

  % Now derive hidden probabilities again for the reconstructed visible sample, then turn these
  % into an actual hidden state sample.
  hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_sample);
  % Implementing Question 7 means we don't actually need reconstructed hidden state, we can just
  % use the hidden state probabilities.
  %hidden_sample = sample_bernoulli(hidden_probabilities);

  % Compute the CD 1 configuration goodness gradient using these actual visible and hidden states
  % now.
  % Implementing Question 7 means we don't actually need reconstructed hidden state, we can just
  % use the hidden state probabilities.
  %reconstruction_goodness = configuration_goodness_gradient(visible_sample, hidden_sample);
  reconstruction_goodness = configuration_goodness_gradient(visible_sample, hidden_probabilities);

  ret = data_goodness .- reconstruction_goodness;

end
