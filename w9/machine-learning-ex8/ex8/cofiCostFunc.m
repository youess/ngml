function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%




nm = size(Y, 1);
nu = size(Y, 2);
nf = size(X, 2);

%size(Y)          % 5 x 4
%size(X)          % 5 x 3
%size(Theta)      % 4 x 3
%size(X_grad)     % 5 x 3
%size(Theta_grad) % 4 x 3

% compute the cost
%for i = 1:nm
%  for j = 1:nu
%    if R(i, j) == 1
%      J += .5 * (X(i, :) * Theta(j, :)' - Y(i, j)) .^ 2;
%    end
%  end
%end
%J += .5 * lambda * (sum((Theta.^2)(:)) + sum((X .^ 2)(:)));

J = .5 * sum(sum(((X * Theta' - Y).^2) .* R)) + .5 * lambda * (sum((Theta.^2)(:)) + sum((X .^ 2)(:)));

% compute the X grad, sum over user
%for i = 1:nm
%  for k = 1:nf
%    tmp = 0;
%    for j = 1:nu
%      if R(i, j) == 1
%        tmp += (X(i, :) * Theta(j, :)' - Y(i, j)) * Theta(j, k);
%      end
%    end
%    tmp += lambda * X(i, k);
%    X_grad(i, k) = tmp;
%  end
%end
for i = 1:nm
  idx = find(R(i, :) == 1);
  Theta_tmp = Theta(idx, :);
  Y_tmp = Y(i, idx);
  X_grad(i, :) = (X(i, :) * Theta_tmp' - Y_tmp) * Theta_tmp + lambda * X(i, :);
end

% compute the Theta grad
%for j = 1:nu
%  for k = 1:nf
%    tmp = 0;
%    for i = 1:nm
%      if R(i, j) == 1
%        tmp += (X(i, :) * Theta(j, :)' - Y(i, j)) * X(i, k);
%      end
%    end
%    tmp += lambda * Theta(j, k);
%    Theta_grad(j, k) = tmp;
%  end
%end
for j = 1:nu
  idx = find(R(:, j) == 1);
  X_tmp = X(idx, :);
  Y_tmp = Y(idx, j);
  Theta_grad(j, :) = (X_tmp * Theta(j, :)' - Y_tmp)' * X_tmp + lambda * Theta(j, :);
end



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
