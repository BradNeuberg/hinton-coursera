function hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state)
  % <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
  % <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
  % The returned value is a matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
  % This takes in the (binary) states of the visible units, and returns the activation probabilities of the hidden units conditional on those states.
  m = size(visible_state, 2);
  number_hidden = size(rbm_w, 1);
  number_visible = size(visible_state, 1);
  hidden_probability = zeros(number_hidden, m);

  for k = 1:m
    visible_state_k = visible_state(:, k);
    hidden_probability(:, k) = 1 ./ (1 + exp(-(rbm_w * visible_state_k)));
  end
end
