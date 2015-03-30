function G = configuration_goodness(rbm_w, visible_state, hidden_state)
  % <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
  % <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
  % <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
  % This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
  m = size(hidden_state, 2);
  number_hidden = size(rbm_w, 1);
  number_visible = size(rbm_w, 2);
  G = 0;

  for k = 1:m
    for i = 1:number_visible
      for j = 1:number_hidden
        G += visible_state(i, k) .* rbm_w(j, i) .* hidden_state(j, k);
      end
    end
  end

  G = G / m;
end
