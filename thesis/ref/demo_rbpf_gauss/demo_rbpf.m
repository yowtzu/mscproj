% PURPOSE : PF and RBPF for conditionally Gaussian JMLS.

% COPYRIGHT : Nando de Freitas    
% DATE      : June 2001

clear;
echo off;

% =======================================================================
%              INITIALISATION AND PARAMETERS
% =======================================================================

N = 100;                     % Number of particles.
T = 50;                     % Number of time steps.

% Here, we give you the choice to try three different types of
% resampling algorithms: multinomial (select 3), residual (1) and 
% deterministic (2). Note that the code for these O(N) algorithms is generic.

resamplingScheme = 2;    

n_x = 1;                    % Continuous state dimension.
n_z = 5;                    % Number of discrete states.
n_y = 1;                    % Dimension of observations.

par.A = zeros(n_x,n_x,n_z);
par.B = zeros(n_x,n_x,n_z);
par.C = zeros(n_y,n_x,n_z);
par.D = zeros(n_y,n_y,n_z);
par.E = zeros(n_x,n_x,n_z);
par.F = zeros(n_x,1,n_z);
par.G = zeros(n_y,1,n_z);
for i=1:n_z,
  par.A(:,:,i) = i*randn(n_x,n_x);
  par.C(:,:,i) = i*randn(n_y,n_x);
  par.B(:,:,i) = 0.01*eye(n_x,n_x);    
  par.D(:,:,i) = 0.01*eye(n_y,n_y);    
  par.F(:,:,i) = (1/n_x)*zeros(n_x,1);
  par.G(:,:,i) = (1/n_y)*zeros(n_y,1);   
end;
%%
par.T = unidrnd(10,n_z,n_z);           % Transition matrix.
for i=1:n_z,
  par.T(i,:) = par.T(i,:)./sum(par.T(i,:)); 
end;
%%
par.pz0 = unidrnd(10,n_z,1);            % Initial discrete distribution. 
par.pz0 = par.pz0./sum(par.pz0); 
par.mu0 = zeros(n_x,1);                 % Initial Gaussian mean.
par.S0  = 0.1*eye(n_x,n_x);             % Initial Gaussian covariance.  

%%

% I sometimes set some of the following matrices by hand:
%par.T = [.1 .9; 
%	 .1 .9];
%par.pz0 = [.5 .5]';   

%par.T = [.1 .5 .4; 
%	 .1 .6 .3
%	 .1 .3 .6];
%par.pz0 = [.5 .5 .5]';   



% =======================================================================
%                          GENERATE THE DATA
% =======================================================================

x = zeros(n_x,T);
z = zeros(1,T);
y = zeros(n_y,T);
u = zeros(1,T);           % Control signals.
%%
x(:,1) = par.mu0 + sqrtm(par.S0)*randn(n_x,1);
z(1) = length(find(cumsum(par.pz0')<rand))+1;

for t=2:T,
  z(t) = length(find(cumsum(par.T(z(t-1),:)')<rand))+1;
  x(:,t) = par.A(:,:,z(t))*x(:,t-1) + par.B(:,:,z(t))*randn(n_x,1) + par.F(:,:,z(t))*u(:,t); 
  y(:,t) = par.C(:,:,z(t))*x(:,t) + par.D(:,:,z(t))*randn(n_y,1) + par.G(:,:,z(t))*u(:,t);  %u is just an atom
end;

figure(1)
clf
subplot(311)
plot(1:T,z,'r','linewidth',2);
ylabel('z_t','fontsize',15);
axis([0 T+1 0 n_z+1])
grid on;
subplot(312)
plot(1:T,x,'r','linewidth',2);
ylabel('x_t','fontsize',15);
grid on;
subplot(313)
plot(1:T,y,'r','linewidth',2);
ylabel('y_t','fontsize',15);
xlabel('t','fontsize',15);
grid on;
%%
fprintf('\n')
fprintf('\n')
fprintf('Estimation has started')
fprintf('\n')

%%
% =======================================================================
%                              PF ESTIMATION
% =======================================================================


% INITIALISATION:
% ==============
z_pf = ones(1,T,N);            % These are the particles for the estimate
                               % of z. Note that there's no need to store
                               % them for all t. We're only doing this to
                               % show you all the nice plots at the end.
z_pf_pred = ones(1,T,N);       % One-step-ahead predicted values of z.
x_pf = 10*randn(n_x,T,N);      % These are the particles for the estimate x.
x_pf_pred = x_pf;  
y_pred = 10*randn(n_y,T,N);    % One-step-ahead predicted values of y.
w = ones(T,N);                 % Importance weights.
initz = 1/n_z*ones(1,n_z);     
for i=1:N,
  z_pf(:,1,i) = length(find(cumsum(initz')<rand))+1; 
end;
%%
disp(' ');
tic;                  % Initialize timer for benchmarking
%%
for t=2:T,    
    
  fprintf('PF :  t = %i / %i  \r',t,T); fprintf('\n');  

  % SEQUENTIAL IMPORTANCE SAMPLING STEP:
  % =================================== 
  for i=1:N,
    % sample z(t)~p(z(t)|z(t-1))
    z_pf_pred(1,t,i) = length(find(cumsum(par.T(z_pf(1,t-1,i),:)')<rand))+1;
    % sample x(t)~p(x(t)|z(t|t-1),x(t-1))
    x_pf_pred(:,t,i) = par.A(:,:,z_pf_pred(1,t,i)) * x_pf(:,t-1,i) + ...
                       par.B(:,:,z_pf_pred(1,t,i))*randn(n_x,1) + ...
                       par.F(:,:,z_pf_pred(1,t,i))*u(:,t); 
  end;

  % Evaluate importance weights.
  for i=1:N,
    y_pred(:,t,i) =  par.C(:,:,z_pf_pred(1,t,i)) * x_pf_pred(:,t,i) + ...
                     par.G(:,:,z_pf_pred(1,t,i))*u(:,t); 
    Cov = par.D(:,:,z_pf_pred(1,t,i))*par.D(:,:,z_pf_pred(1,t,i))'; 
    w(t,i) =  (det(Cov)^(-0.5))*exp(-0.5*(y(:,t)-y_pred(:,t,i))'* ...
				    pinv(Cov)*(y(:,t)-y_pred(:,t,i))) + 1e-99;
  end;  
  w(t,:) = w(t,:)./sum(w(t,:));       % Normalise the weights.

  
  % SELECTION STEP:
  % ===============
  if resamplingScheme == 1
    outIndex = residualR(1:N,w(t,:)');        % Higuchi and Liu.
  elseif resamplingScheme == 2
    outIndex = deterministicR(1:N,w(t,:)');   % Kitagawa.
  else  
    outIndex = multinomialR(1:N,w(t,:)');     % Ripley, Gordon, etc.  
  end;
  z_pf(1,t,:) = z_pf_pred(1,t,outIndex);
  x_pf(:,t,:) = x_pf_pred(:,t,outIndex);

end;   % End of t loop.

time_pf = toc;     % How long did this take?


%%

% =======================================================================
%                              RBPF ESTIMATION
% =======================================================================


% INITIALISATION:
% ==============
z_rbpf = ones(1,T,N);          % These are the particles for the estimate
                               % of z. Note that there's no need to store
                               % them for all t. We're only doing this to
                               % show you all the nice plots at the end.
z_rbpf_pred = ones(1,T,N);     % One-step-ahead predicted values of z.
mu = 0.01*randn(n_x,T,N);      % Kalman mean of x.
mu_pred = 0.01*randn(n_x,N); 
Sigma = zeros(n_x,n_x,N);      % Kalman covariance of x.
Sigma_pred = zeros(n_x,n_x,N); 
S = zeros(n_y,n_y,N);          % Kalman predictive covariance.
y_pred = 0.01*randn(n_y,T,N);  % One-step-ahead predicted values of y.
w = ones(T,N);                 % Importance weights.
initz = 1/n_z*ones(1,n_z);     
for i=1:N,
  Sigma(:,:,i) = 1*eye(n_x,n_x); 
  Sigma_pred(:,:,i) = Sigma(:,:,i);
  z_rbpf(:,1,i) = length(find(cumsum(initz')<rand))+1; 
  S(:,:,i) = par.C(:,:,z_rbpf(1,1,i))*Sigma_pred(:,:,i)*par.C(:,:,z_rbpf(1,1,i))' + ...
                par.D(:,:,z_rbpf(1,1,i))*par.D(:,:,z_rbpf(1,1,i))';
end;

disp(' ');
tic;                  % Initialize timer for benchmarking

for t=2:T,    
  fprintf('RBPF :  t = %i / %i  \r',t,T); fprintf('\n');  

  % SEQUENTIAL IMPORTANCE SAMPLING STEP:
  % =================================== 
  for i=1:N,
    % sample z(t)~p(z(t)|z(t-1))
    z_rbpf_pred(1,t,i) = length(find(cumsum(par.T(z_rbpf(1,t-1,i),:)')<rand))+1;
    
    % Kalman prediction:
    mu_pred(:,i) = par.A(:,:,z_rbpf_pred(1,t,i))*mu(:,t-1,i) + ... 
                   par.F(:,:,z_rbpf_pred(1,t,i))*u(:,t); 
    Sigma_pred(:,:,i)=par.A(:,:,z_rbpf_pred(1,t,i))*Sigma(:,:,i)*par.A(:,:,z_rbpf_pred(1,t,i))'...
                      + par.B(:,:,z_rbpf_pred(1,t,i))*par.B(:,:,z_rbpf_pred(1,t,i))'; 
    S(:,:,i)= par.C(:,:,z_rbpf_pred(1,t,i))*Sigma_pred(:,:,i)*par.C(:,:,z_rbpf_pred(1,t,i))' + ...
              par.D(:,:,z_rbpf_pred(1,t,i))*par.D(:,:,z_rbpf_pred(1,t,i))';  
    y_pred(:,t,i) = par.C(:,:,z_rbpf_pred(1,t,i))*mu_pred(:,i) + ... 
                      par.G(:,:,z_rbpf_pred(1,t,i))*u(:,t);
  end;
  % Evaluate importance weights.
  for i=1:N,
    w(t,i) = (det(S(:,:,i))^(-0.5))*  ...
             exp(-0.5*(y(:,t)-y_pred(:,t,i))'*pinv(S(:,:,i))*(y(:,t)- ...
						  y_pred(:,t,i))); 
  end;  
%  w(t,:) = exp(log_w(t,:))+ 1e-99*ones(size(w(t,:)));
  w(t,:) = w(t,:)./sum(w(t,:));       % Normalise the weights.

  
  % SELECTION STEP:
  % ===============
  if resamplingScheme == 1
    outIndex = residualR(1:N,w(t,:)');        % Higuchi and Liu.
  elseif resamplingScheme == 2
    outIndex = deterministicR(1:N,w(t,:)');   % Kitagawa.
  else  
    outIndex = multinomialR(1:N,w(t,:)');     % Ripley, Gordon, etc.  
  end;
  z_rbpf(1,t,:) = z_rbpf_pred(1,t,outIndex);
  mu_pred = mu_pred(:,outIndex);
  Sigma_pred = Sigma_pred(:,:,outIndex);
  S = S(:,:,outIndex);
  y_pred(:,t,:) = y_pred(:,t,outIndex);


  % UPDATING STEP:
  % ==============
  for i=1:N,
    % Kalman update:
    K = Sigma_pred(:,:,i)*par.C(:,:,z_rbpf(1,t,i))'*pinv(S(:,:,i));
    mu(:,t,i) = mu_pred(:,i) + K*(y(:,t)-y_pred(:,t,i));
    Sigma(:,:,i) = Sigma_pred(:,:,i) - K*par.C(:,:,z_rbpf(1,t,i))*Sigma_pred(:,:,i);   
  end;

end;   % End of t loop.

time_rbpf = toc;     % How long did this take?


% =======================================================================
%                          SUMMARIES AND PLOTS
% =======================================================================


z_plot_pf = zeros(T,N);
z_plot_rbpf = zeros(T,N);
for t=1:T,
  z_plot_pf(t,:) = z_pf(1,t,:);
  z_plot_rbpf(t,:) = z_rbpf(1,t,:);
end;

z_num_pf = zeros(T,n_z);
z_num_rbpf = zeros(T,n_z);
z_max_pf = zeros(T,1);
z_max_rbpf = zeros(T,1);
for t=1:T,
  for i=1:n_z,
    z_num_pf(t,i)= length(find(z_plot_pf(t,:)==i));
    z_num_rbpf(t,i)= length(find(z_plot_rbpf(t,:)==i));
  end;
  [arb,z_max_pf(t)] = max(z_num_pf(t,:));  
  [arb,z_max_rbpf(t)] = max(z_num_rbpf(t,:));  
end;

figure(2) 
clf
plot(1:T,z,'k',1:T,z,'ko',1:T,z_max_rbpf,'r+',1:T,z_max_pf,'bv','linewidth',1);
legend('','True state','RBPF MAP estimate','PF MAP estimate');
axis([0 T+1 0.5 n_z+0.5])


detect_error_pf   = sum(z~=z_max_pf');
detect_error_rbpf = sum(z~=z_max_rbpf');


disp(' ');
disp('Detection errors');
disp('-----------------------------');
disp(' ');
disp(['PF      = ' num2str(detect_error_pf)]);
disp(['RBPF    = ' num2str(detect_error_rbpf)]);
disp(' ');
disp(' ');
disp('Execution time  (seconds)');
disp('-------------------------');
disp(' ');
disp(['PF      = ' num2str(time_pf)]);
disp(['RBPF    = ' num2str(time_rbpf)]);
disp(' ');


figure(3)
clf;
domain = zeros(N,1);
range = zeros(N,1);
thex=[0.5:0.05:n_z+.5];
hold on
ylabel('t','fontsize',15)
zlabel('Pr(z_t|y_{1:t})','fontsize',15)
xlabel('z_t','fontsize',15)
for t=1:1:T,
  [range,domain]=hist(z_plot_rbpf(t,:)',thex);
  waterfall(domain,t,range/sum(range))
end;
view(-30,80);
rotate3d on;
a=get(gca);
set(gca,'ygrid','off');
title('RBPF')



