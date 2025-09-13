---
layout: post
title: Unsupervised Learning, K-Means and Expectation Maximisation algorithm.
tags: [Machine Learning, cs229, Algorithms] 
---

Reference: [Lecture from CS229 Lecture 17 by Anand Avati](https://www.youtube.com/watch?v=LmpkKwsyQj4)

## Agenda:
1. Introduction to Unsupervised Learning
   - Contrast with supervised learning and reinforcement learning
   - Goal: Find interesting structures in data without labels
   - Examples: Clustering, density estimation

2. K-Means Algorithm
   - Purpose: Grouping data into K clusters
   - Steps:
     a. Initialize cluster centroids randomly
     b. Assign each point to the nearest centroid
     c. Recalculate centroids based on assignments
     d. Repeat until convergence
   - Discussion on convergence and local optima

3. Gaussian Mixture Models (GMM)
   - Extension of K-means with probabilistic assignments
   - Model components:
     - Latent variable Z (cluster assignment)
     - Observed variable X (data point)
     - Parameters: means, covariances, and mixing coefficients
   - Soft assignments instead of hard assignments

4. Expectation Maximization (EM) Algorithm
   - General framework for maximum likelihood estimation with latent variables
   - Two main steps:
     a. E-step: Compute posterior probabilities (soft assignments)
     b. M-step: Update parameters to maximize the likelihood
   - Application to GMM

5. Mathematical Foundations
   - Jensen's Inequality
     - Definition for convex and concave functions
     - Application in deriving the EM algorithm
   - Evidence Lower Bound (ELBO)
     - Definition and significance in EM
   - Derivation of EM algorithm using Jensen's inequality

6. Convergence of EM Algorithm
   - Intuitive explanation using graphical representation
   - Brief mention of formal proof (to be covered in next lecture)

7. Practical Applications
   - Brief mention of market segmentation as an example

8. Relationship to Deep Generative Models
   - Importance of understanding EM for modern machine learning techniques
   - Mention of Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs)


## Full Transcript:

All right, welcome back everyone. Hope you had a good weekend. So this is lecture 16 of CS229, and today we're going to start a new chapter on unsupervised learning. 

Unsupervised learning will be the broad topic for the rest of this week and parts of next week, and the specific topics for today are the k-means algorithm, mixture of Gaussians (which is also called the GMM or Gaussian Mixture Models), and the expectation maximization algorithm.

To jump into today's topics, what we've seen so far in the first few weeks (first three or four weeks or so) we covered supervised learning where we were trying to learn a function that maps X to Y. We were given pairs of X, Y as our training set or examples. After supervised learning, we went into some learning theory, we studied bias variance trade-off in the bias variance analysis, and saw a little bit into generalization. 

Then last week we saw reinforcement learning where the goal, rather than minimizing some kind of a loss function, we want to maximize the value by choosing a suitable policy, right? Here value is the long term cumulative sum of discounted rewards, and we want to maximize the long term reward by choosing a policy.

Now, the new chapter we are going to start today (starting today) is unsupervised learning. In the unsupervised learning case, we are given a data set – generally a collection of X1, X2, etcetera, Xn, right – and we do not have a corresponding Y variable associated with each X variable. We are just given a set of collection examples, a set of X's, where each Xi is in Rd – in a d dimensional dual space, for example. Our goal is to learn some kind of a structure in these X's, right? 

We do not have a corresponding correct answer. We do not have what's otherwise called supervision of what the correct answer is for each X. But in general, we are just given some collection of X's, and our goal is to find some kind of an interesting structure in these X's that hopefully gives us some new insight, right?

So, we have seen logistic regression before, and in the case of logistic regression, you know,  X1 XD…  let me use a few colors here. We were given a data set like this, and our goal was to find a separating hyperplane. This is supervised learning because we are given the correct answer (that is, the color of each point) along with the point itself, right?

Whereas in the unsupervised learning, the problem would translate to something like this: we are given some points – just the X's – and our goal now is to learn some kind of an interesting structure here. Previously, we had the correct answer for each input. Now, we are just given a collection of X's and a reasonable structure to find in this are these two clusters. 

So loosely speaking, we want to look for such structures when we are given just X's. However, this problem is generally not very well defined. In the first case, for each point we were told what the correct answer is. But consider a problem like this. If you are asked to find an interesting substructure in this kind of data, it would be totally reasonable to say this is one cluster, this is one cluster, and this is another cluster, right? Another totally reasonable thing would be to say this is one cluster, and this is one cluster, right? So in a way there is no correct answer, so to speak, and our goal is to learn some kind of an interesting structure in the presence of such kind of ambiguities.

The way you want to think of this is classification problems in the supervised setting are somewhat related to clustering problems in the unsupervised setting, where the cluster identity is like the class label. We want to, looking at just the X's, find out both how many number of classes there are and also to which class each example belongs to, right?

And why would this be interesting? This would be interesting, for example, in examples where, supposing you have, you're working at a marketing department and you have information about your customers. The information about your customers can be represented in, you know, in some kind of a vector space where you know you have the age of the customer here, you have, you know, their annual income, and on another axis you might have their, you know, geographical location. Each customer would be a point in the space, right?

Now as a person who is working in marketing, you might be interested to perform some kind of a market segmentation to identify, you know, groups of customers so that you can do some kind of a targeted, you know, advertising or marketing campaign or some such thing, right? So that's just one example of why unsupervised learning might be, you know, interesting.

So the first unsupervised learning algorithm that will be seeing is something called the k-means clustering algorithm. The k-means clustering algorithm is pretty straightforward. This is probably one of the simplest algorithms for unsupervised learning.

**K-means Algorithm**

So we are given a data set of n examples: X1 through Xn where each Xi is in Rd. Our goal is to group them into k clusters. For the purpose of the algorithm, we are, we will assume that k, n is given to us. The algorithm goes like this:

Initialize cluster centroids mu1, mu2, muk – so you have one centroid per cluster – where each of them are in Rd, randomly. So each of these mu1, mu2, mu k is a vector. Previously, in our notations, generally having a suffix to a variable generally meant it was a scalar. But in this case, mu1 is a full vector, right, and there are k such full vectors:  mu1 through muk, and we initialize them randomly.

And then we repeat this until convergence. Repeat until convergence.

For every i, where i denotes the example number,  set Ci equal to argmin j  Xi minus mu j square.

And then, that's step one.

Step two: for every j, where j is now, where j indicates the cluster identity, set mu j is equal to sum over i equals one to n indicator of  Ci equals j of Xi and sum over i equals 1 to n indicator Ci equals j.

So what are we doing here? 

K-means is an iterative algorithm where we are given a set of n examples, which we index by i, and we want to identify k clusters, where the k clusters are indexed by j. We initialize the cluster centroids randomly, where mu1 through muk are each a vector in Rd, and we repeat until convergence. 

Where for every, for every i, we set Ci… here C you can think of C as an array of length n where for each example there is a corresponding Ci.  For every Xi, there is a corresponding Ci. And we set Ci to be the identity, that's the argument of j, identity of the nearest mean.

Based on the set Ci vector or Ci array, we then recalculate mu j, where mu j is calculated as the mean of all Xi's for which Ci equals j.

Question. So as I said already, for now let's assume k is told to us, you know, we are given what k is. And this is the algorithm, right? The way, it's a pretty straightforward algorithm where we, where we alternate between one step where we are either where we are calculating the cluster identities for each example, and in the other step where we are recalculating the cluster centroids. 

This is probably seen through a simple visualization which we'll have a quick look at.

Any questions on this so far? 

Yes.

Questions, so the question is, what happens… can we use an unsupervised learning setting to learn the different cluster centers and use that as a classification algorithm? Yeah, it might or it might not behave the same way as a supervised learning algorithm.

Yeah, so supposing this is, you know, think of this as a collection of points that are given to us, where each green point is, you know, is a data point in,  Xi in Rd. The way the algorithm works like this: we, here we assume k equals 2, and the red X and the blue X you can think of them as mu1 and mu2, which are randomly initialized.

In the first step, what we do is for each point, for each point X, we identify the nearest cluster. We, we set Ci to be the identity of that cluster which has the smallest L2 distance between that point and the cluster centroid. So over here, the red dots are those for which the red X is closer, and the blue dots are those X's for which the blue X is closer, right? So this is like setting the Ci's in the first iteration.

Once we set the Ci's, in the next step what we do is recalculate the mu, the mu j's. So what happened here? I'm going to go back one slide just to see the difference. 

So these two points which previously belong to the, the old blue centroid now got mapped to the new, to the red one. 

![[Pasted image 20240814121147.png]]

And then we re-evaluate the centroids again, and the centroids will now move to the center. Once we reach here, in the next iteration I actually moved a slide, you know, nothing changes and we declare that the algorithm has converged, right? 

It's a pretty, pretty simple and straightforward algorithm. 

Now, the, a few natural questions to ask is, will this algorithm always converge? And will it always give us the same, same answers all the time? So it can be shown that the algorithm does always converge. What we mean by convergence in this algorithm has a special meaning. 

So if we consider this loss function called J of C comma mu to be equal to i equals 1 through n  Xi minus mu Ci square.

![[Pasted image 20240814122246.png]]

This is also called the distortion function. The k-means algorithm is basically an algorithm to minimizing this particular loss or the distortion function in the form of coordinate descent. 

So what is coordinate descent? You can think of coordinate descent as a variant of gradient descent where what we are doing is at each step, instead of minimizing the loss with respect to all the input variables, we only minimize the loss with respect to a few variables by holding the others fixed, right?

So the, step number one corresponds to minimizing the distortion function by holding mu fixed and optimizing C, where we calculate new C's. Step number two corresponds to then minimizing J again by holding the C's fixed and optimizing it with respect to mu, right? So k-means is coordinate descent of the distortion function J where in one step we optimize it with respect to C, in the other step we optimize it with respect to mu, and the result of the optimization results in these, in these closed form rules for recalculating the C's and mus.

We say that k-means algorithm converges in the sense that eventually we are going to reach some kind of a local minima of this J function. It may so happen that we may have minimized J, but we may end up toggling between two sets of me or two sets of mus and C's, alternating once we reach a local minima. Though that happens extremely rarely in practice, but we will eventually reach a state where J is no longer minimized further. We're going to flatten out in J, and most of the times, pretty much all the time in practice, that's going to result in some kind of mu and C that does not change anymore. This J is non-convex, which means the mus and the C's that we end up in can change from run to run. If you start with a different initialization, you may end up with a different set of mus and C's, right, which kind of ties back to a question that was asked before.

You know, why do we, you know, ever need to use the label identities and not just perform, perform clustering? The answer is is that this is basically a non-convex problem, and we can end up with different, different cluster identities depending on the initialization. Any questions on this?

Yes. 

Question. So the question is, by looking at a function, how do we determine whether it is convex or not? In general it is… the answer is not always straightforward. So it is easy to show that something is convex by showing it as, you know, a composition of convex sub functions, right? However showing something is not convex is not always that straightforward. Something which may not appear to be convex at first can sometimes with some kind of reparameterization may end up being convex, etcetera. In this case, it happens to be non-convex. Any questions on this? 

Cool. 

So given this clustering pro, clustering approach that we have seen, let us move on to something that is slightly different and also somewhat related, which is the problem of density estimation.

So density estimation generally refers to the problem where we are given some number of data points, given some number of data points… and this is in R, you know, think of it as the X axis, and these points are residing in a continuous space, right… and the goal is to now… we assume that these points are sampled from some kind of a probability distribution. Because this, this, these points are coming on a, on a continuous distribution, the corresponding probability distribution has some kind of a density. It is not a probability mass function, but it is a probability density function, right?

Given these points, points that look like this, the question is, what is the density function from which these points were sampled? In general, it's a very hard problem because if you want to fit this data really, really well then the best possible fit would be a density that looks like this, where it has, you know, like a Dirac delta function over every data point. You know, a peak that, you know, that's really, really peaked over each data point. You can, you know, this, this, this, this is a valid, valid density, but at the same time it does not feel natural, right?

Another, another equally valid density would be something that looks like this.

Now this is also a valid density from which it could have been sampled because there is nothing to the left, nothing to the right, and there are some values over here, you know, that there are something where there's some data. So you might, you might have some kind of a density. Also this is also valid, right? 

So all these are different possible answers for what the underlying density is from which these points are calculated. Kind of the fundamental problem of density estimation is that the density function has to be a continuous function. 

In the case, if these were, you know, outcomes of coin tosses where the support was discrete, then maximum likelihood was, was pretty straightforward. You could, you know, treat them as a multinomial and just count them. But whereas in density estimation, we need to come up with a smooth function over discrete observations.

I say discrete observation because we have a fixed number of observations and we want to come up with, with a smooth estimate. So a common approach in density estimation is to use this model called the Gaussian Mixture Model, or it is also called the mixture of Gaussians, right? 

The Gaussian Mixture Model… where, given some, given some data points that look like this, right, we want to, we make this hypothesis that there are these two underlying, different, two different distinct Gaussian distributions. There is one Gaussian distribution from which these were sampled, another Gaussian distribution from which these were sampled. Together you can take the sum of these two Gaussians, Gaussian probability distributions and say this entire data set is sampled from these two different Gaussians, these two mixtures of these two Gaussians.

![[Pasted image 20240814131117.png]]

The choice of k is something again, is, is similar to, you know, in k-means. It, it is something that we choose by, by, you know, visual inspection or, or, or in general seeing how well the data fit the, fit the number of k. 

The problem we have now is to, is to, given this data set, estimate the two Gaussians from which the data set might have come, right? We are not told what the identity of the two Gaussians are. If this were to be a supervised setting, then we would have, you know, right, they would come with some kind of an identity and we could have fit one Gaussian here and the other here. 

This is exactly what we did in… do any of you remember where we did something like this?

GDA. 

Exactly. So in GDA, we were, we were told that X's come from, are sampled from Gaussians, and there are these two different classes: class 1 and class 2, right? And our goal was to take these X's along with their cluster identities (the corresponding Y's) and estimate the mus and sigmas for the two classes, right?

Now in Gaussian Mixture Model, we are essentially generalizing GDA in a way where we are not given what the Y labels are. We are just given the X's, and we also relax the constraint which we had in GDA that the covariance had to be the same. In this case, the covariance can be different, and our goal is to now come up with some kind of a density p of X that allows us to, assign probability density to the observed values, right? So that is the, that is the setting in which the Gaussian Mixture Model comes into picture. 

The reason why, why we are interested to even calculate this p of X… there are many reasons why calculating p of X could be interesting. 

So here's one example. Here's a completely made up example. Supposing, supposing you are an aircraft manufacturer where let's assume the, the parts that we manufacture have two kind of attributes. Let's say, you know, heat tolerance… and if any of you are, you know, are in aeronautics, what, what might be another, you know… let's say, let's, let's call it,  heat tolerance and, and, whatever… power output, whatever that means. 

So let's assume there are, there are, there are these two kind of, attributes that are, that are, therefore, some part that we manufacture. In general, what we observe, that if we were to plot the, the, the every manufactured part as a point here, we might observe that most of the normal parts fall along some kind of a distribution like this. Maybe there are, you know, two different kinds of, subtypes of parts may be based on the material or something, where some of them fall in distribution, some of them fall in this kind of a distribution, you know, whatever be the reason, but generally, let's assume that, you know, normal looking parts fall in this kind of, belong to this kind of a probability distribution. 

Now suppose we want to, we want to have some kind of an automated anomaly detection where we want to detect that, you know, some part is, is faulty for some reason. For example, a part that may, that has this attribute, right? We want to identify that this point over here is, is faulty. At first, it, it might appear, you know, even though this looks visually away from this kind of a heat map, if you were to look at any one of the axes alone, it looks pretty normal. 

From the heat tolerance point of view, it's kind of, you know, in the close to the mean. If you just look at the power output, it is also kind of near the mean. But it is this combination that makes it kind of, you know, an anomalous example, right? 

The way this kind of an anomaly detection is, can be carried out in practice is to construct, you know, a density estimate p of X for these points, where the, where this p of X assigns high probability for anything that falls in this region, and p of X assigns low probability for anything outside this region, right? 

A common approach to doing this kind of a density estimation is to use mixture of Gaussians. The way we go about doing mixture of Gaussians is… so first what we are going to do is to provide you an algorithm to do mixture of Gaussians, and we are going to provide, construct this first algorithm based purely on intuition. Then what we are going to do is describe this general framework called expectation maximization and then re-derive Gaussian mixture, the Gaussian Mixture Model using this framework, and see that we end up with the same algorithm that we got using intuition. 

The expectation, maximization that we're going to see next is a more general framework that works for a broad class of, of generative models. These are examples of generative models, and the Gaussian Mixture Model is just one such, one such model which can be solved through expectation maximization. This question.

Restrict like… what is the upper bound for the frequency that you will tolerate on the PDF because the k will fit best as the k, as k approaches the number of data points, right?

So the, the question is, I guess to kind of summarize this, how do we find out k? It is true that as we increase the number k, we kind of fit the data better and better. In order to kind of think of what's the best value of k, I will leave this as a thought exercise for now and we'll come back to this probably next week. Try to see how you can apply, you know, what we learned in, in learning theory, bias and variance, you know, how, how can you, you know, give some thought on how you can apply bias variance analysis on this kind of a problem, right? We will come back to this later. For now, we'll, you know, for, for, for today and for, the rest of this week, we're going to just cover more algorithms, and we'll you know, come back to it later with, you know, and see how we can kind of approach it in a more principled way. You know, for now, you know, as a mental exercise, see how you can apply bias variance analysis in this kind of a setting, right?

So in the mixture of Gaussians or Gaussian Mixture Model… so we are, we are given a training set of just X's, right? And then we are going to assume there is this Zi which belongs to a multinomial distribution, parameter with parameter phi, where vj is greater than equal to 0 and the sum over all j equals 1 through k vj equals 1, right? Phi j is basically… vj is the probability that Zi equals j, right?

And then we have Xi given Zi equals j to be distributed from a normal distribution of mean mu j and covariance sigma j, right?

So this is describing the model. The, the way we assume the model works is that first we sample the class identity Z from some kind of a multinomial distribution, right? Then one, you know, once we, once we have a sample, the identity we generate, you know, an observation X condition on the Z value that we sampled from what, some particular Gaussian distribution with mean mu j and, and covariance, sigma j. 

This is very similar to GDA, right? In GDA, the, the differences between this and GDA is that in GDA, we called Z as Y. Here, we're just calling it Z, and that is a common pattern that you will see when you know, in, in the algorithms where we are doing full observation which becomes a supervised setting. **The variables that we call Y we end up calling them as Z in unsupervised setting because they are not observed**. In this case, there is this underlying Z that we do not observe that is sampled from some multinomial distribution. Depending on the identity of the cluster that we, that we sample, the observation is then sampled from a corresponding Gaussian distribution that has a mean and covariance specific to that cluster, right? 

**In these cases, Zi's, because they are not observed, are called latent variables. So latent variable is a fancy name for a random variable that we have not observed.** We've just not seen what, what, what its value is, and that's why we call it latent. 

Yes. 

Question. 

So fee over here is the… you, you can think of it as the class prior. You know, the class prior that we had in GDA. This just tells us, of all the examples, X's that we have, what fraction of them belong to cluster j. 

Good question. 

And now in GDA, we performed maximum likelihood estimation in GDA, and our maximum likelihood objective was… in GDA, it was log p of X comma Y where mu sigma and p were the parameters, right? So this was the log likelihood objective in GDA, right?

Whereas in Gaussian Mixture Model, we do not observe Y, right? And so in the, in the Gaussian Mixture Model, our objective will be to maximize log p of X given, for, for phi mu sigma, and this will be our likelihood function. That's the only difference.

Over here, the objective was the full joint distribution. Over here we would have liked to do the same, but we haven't observed the work, the corresponding Y's – we should call as Z's here, right – there they're not observed. So we cannot construct a likelihood function because we won't know what value of Z to plug in in this expression, right? If you had observed them, it was pretty straight forward. We would have been just doing GDA. 

Instead, what we do is maximize log p of X. Log p of X can, that can also be written as log sum over Z p of X comma Z phi mu, right? So we write out the full joint distribution and marginalize out the latent variable. 

![[Pasted image 20240814140728.png]]
Any questions how we went from this to this?

This question. So the question is, shouldn't Z also contribute to our likelihood objective? The answer is, if we had observed Z, then yes it should have. But we don't know what Z is, so it you know, it cannot… so the question is, we are assuming there are k clusters. Given a setting, you know, some value for k, shouldn't we therefore account for, for the cluster identity? Are making an assumption about k, but we do not know which of those k clusters each point belongs to, right? So our objective is to now maximize this expression, which is the same as this expression, right?

For the rest of, of, you know, today's lecture and, and throughout, it can be useful to set up, you know, some kind of, terminology. For example, p of Z, we will call it class prior. Or in cases where Z is not discrete, but it is continuous, we'll just call it prior, right?

P of X comma Z, we will call it the model, because it describes the full data generating process. The joint distribution always describes the full data generating process, and that's always called the model, right?

Z in unsupervised settings is called latent because we do not observe it. Latent is just a fancy word for unobserved. 

P of Z given X, we will call it the posterior. 

Finally, p of X… so p of Z is called the prior, and p of X will be called the evidence. Evidence because X is what we observe, that is the evidence based on which we are performing inference, right? This is just terminology, and this terminology is pretty standard and used in many papers, many, many books. 

So our goal is to maximize the likelihood using the evidence.


Right? And this… the way we go about doing that is… right?

So to directly maximize this evidence, directly maximizing this likelihood… if we were to attempt it, the way we did it with GDA of taking derivatives, setting it equal to zero and solving for the parameters… you will observe that you, you won't get a close form expression for this. You can try it out. You know, you won't get a close form expression for the way we got it with GDA. In GDA, we got a closed form expression because we had observed both X's and Z's – we called it Y's – and if we had observed both X's and Y's here, we would have gotten a closed form expression. But because the Z's are not observed and we are taking this… marginalizing them… if you work it out, we will not get close form expressions. 

So instead what we will do is, just like, you know, taking inspiration from k-means, we are going to first imagine or, are come up with some kind of an estimate for Zi's first. 

So the algorithm that we are going to do is repeat until convergence, where this is, you know, just taken inspired by k-means. We'll call it the E-step.

For each i comma j, set wij to be equal to p of Zi equals j given Xi, parameters phi mu sigma.

And the L step: update the parameters. 

vj equals 1 over n equals 1 to n w by j. 

mu j is sum over i equals 1 to n wij Xi over i equals 1 to n wij. 

And sigma j is equal to somewhere i equals 1 to n wij Xi minus mu j, Xi minus mu j transpose, divided by i equals 1 to n wij.

So what we did is we start with some random initialization. Randomly initialize parameters. 

The repeat will start after we randomly initialize it. 

**Randomly initialize mu phi and sigma.** 

Start with some random initialization. Think of this as the way we randomly initialize the cluster centroids and k-means, right? Based on the random initialization in k-means, for each point we associated it to the nearest cluster centroid, right? The, the kind of similar operation that we are going to do here is for each point, we are going to assign a weight to each cluster centroid specific to that point, where the weight is the posterior distribution of p of Z given X. So given a point, we calculate a posterior distribution of the probability that this point belongs to a particular centroid, and this is just the posterior. The posterior distribution, we are going to call it a weight. Once we calculate this weight, we are going to reweight all our data points to calculate the corresponding mus and sigmas. 

So for example, Xi will, may belong to, will have a weight of… let us say if there are, if there are say three centroids where k equals 3, then p of Zi equals j given Xi, right? This could be some kind of multinomial distribution like 0.1, 0.7, 0.2, where this belongs to k equals 1, k equals 2, k equals 3. Where if the centroid mu was, was close to, mu2 was close to Xi, then it would have a higher weight, and, you know, these two are farther away, so it has a lower weight. 

In, in case of k-means, we would do a hard assignment of every point to one cluster only. But over here, we are doing a soft assignment where every point has, has a soft assignment in the form of this probability distribution for all the cluster centroids. 

![[Pasted image 20240814143815.png]]

The probability assigned to the centroid that is closer to the point… by closer here we mean in a probabilistic sense where it is, it has a high likelihood in that, in, in that clusters Gaussian distribution, then it will have a high posterior probability… and we do this kind of a soft assignment of every point to, to the set of all clusters. Using the, the calculated weights, we recalculate the mus and sigmas using the weighted data set. Here, every, every point i contributes to every centroid j and the contribution is weighted by the corresponding wij. Questions.

Yeah. So the question is, will this have a closed form expression?

Yeah. Yeah. 

So, so the question is… so for this, I would, I would, remind you in, in case of GDA we had calculated a posterior that was very similar. If you remember, the posterior had a form of… yeah, it had the form of a logistic regression in that case. However we limited ourselves to two, two points… to two classes, and we also had a constraint that sigma was common to all of them. But in this case, when we relax that constraint, what we will observe is that the posterior takes the form of a soft max. 

It takes a form of a soft max that uses quadratic features. Quadratic features of X's. But for a fixed value of, you know, for small k's, you can, you know, come up with, with, with an expression for this. In fact, you'll be doing this in your homework, so that will be clarified in your homework as well, right?

So, so inspired by k-means, here is a version of… you can think of the Gaussian Mixture Model as soft k-means. You know, people call it the soft k-means as well. We call it soft because in the assignment phase in k-means, right, in k-means it was a hard assignment. 

This was… this corresponds to the E-step. The, in k-means, the cluster identity was a hard assignment. You can think of k-means as a way in which we calculate a posterior distribution which is always one hot. 

If, if we calculate these kind of posterior distributions, then we effectively get k-means out of GMMs, right? And in the equivalent of the M step was the way in which we were recalculating mus, the mu j's. We use only those, those X's for which the cluster identity matched the, the corresponding cluster. And over here, instead of, instead of having an indicator function, we are going to use this soft assignment weight. Any questions?

Yeah. So if W's are one hot, then this, this will essentially be, you know,  k-means. 

Yes. 

Question. 

So the question is how do we calculate $w_ij$ if we have not observed the i's, this disease? Is that a question? 

Yeah, so here, in order to calculate this, we don't need to know Zi's, right? So we don't need to… so we are just cons… we are just constructing the probability that Zi could be equal to j, right? The way we go about doing this is to use the Bayes rule, right? The way you use the Bayes rule…


Right, so, and, and this can be calculated as p of, you know… so p of Z given X is equal to p of X given Z times p of Z over sum over all of Z p of X given Z times p of C, right?

Over here, p of X given Z is normal distribution, so this is Gaussian, right? And p of Z is just multinomial, right? In the denominator, you know, it is same… the, the same terms, Gaussian and the multinomial. But you sum it over all the classes, and you will see that, you know, just the way you showed it in homework one for GDA, where in GDA this took the form of a logistic regression, here you can actually show that it takes the form of a soft max. It's very similar calculation. Any questions? 

Right. 

Yeah, so this is basically the Gaussian Mixture Model where we derive the steps to be by taking inspiration from k-means, right? 

We are intentionally giving it the names, the E-step and the N-step because next we're going to talk about the E-M algorithm, and we're going to derive it in a more principled way, and we're going to end up with the same update rules, right? And this is, you know, think of this as soft k-means. 

Soft k-means, right? 

**E-M algorithm.**

Now we are going to switch gears and talk about the E-M algorithm. So the E-M algorithm is also called the expectation maximization algorithm, that is an algorithm where it gives us a framework of maximum of performing maximum likelihood estimation when some variables are unobserved, right?

It is, it, it is used in cases where we have a functional form for the joint p of X comma Z, and X and Z could be anything, but Z's are not observed. 

**So expectation maximization. Expectation maximization is an algorithm where we perform MLE in the presence of latent variables.** 

Where the true model is p of X comma Z with some parameters theta. If we observe X and Z jointly… if everything is observed, then the problem is very simple. We perform simple maximum likelihood estimation. But when Z is unobserved, we want to instead perform maximization of the objective L theta equals log p of X, where Z is marginalized. The EM algorithm or the EM framework gives us a framework for achieving this where we maximize log p of X in an indirect way rather than directly taking the derivatives and setting them to 0 and so on.

Now before we go into the EM algorithm, some kind of… it can be useful to have some kind of a context here. So the EM algorithm was, was discovered sometime in the, I think early 1970s, where, you know, where people were trying to perform MLE in the presence of unobserved, unobserved data. It so happens that this, this framework is so general and so powerful that there are… it has been adapted in so many different ways, and it has been, you know, minor extensions to the EM algorithms are there in so many different forms. But this framework is somewhat central, and understanding EM in a deep way will be extremely useful if you are interested in things like deep generative models.

So in, over the last few years, there has been tremendous growth in deep generative models where you might have heard of, you know, variational autoencoders, generative adversarial networks (or GANs), or flow-based models, glow-based models. Understanding all of them would be a lot, much easier if you really understand the EM algorithm well because the EM framework gives you a kind of a mind map where you can place all these different algorithms and kind of understand the strengths and weaknesses, and you know, what's common between them, what's different between them, and so on. 

So, the EM algorithm is one of the key algorithms to, for, even modern deep learning or deep generative, deep generative models, right? So before we jump into the EM algorithm, we are going to first look at something called Jensen's inequality.

**Jensen's Inequality**

Jensen's inequality is a, is a very general probabilistic inequality that's used, you know, in many places in probability theory and applied probability theory, and it will show up in our derivation of EM algorithm as well. So, you can think of Jensen's inequality as a probabilistic tool that we will use in deriving, uh, EM. But Jensen's inequality by itself is, is a very generic and, you know, commonly used inequality in probability theory. 

So let's, let's assume a function F to be convex. Assume F to be a convex function which means F double prime of X is greater than equal to 0 for all X, right? We say that F is strictly convex if F double prime of X is greater than 0 for all X, right?

So the, the mental picture to have is F of X, and this is X… F of X and X.

So this is an example of a convex function. Convex functions are bowl-shaped functions, and they can have a 0 second derivative in a few places where F prime of X is greater than or equal to 0. But in a strictly convex function, the second derivative is never exactly equal to 0. It is always greater than 0. So if there are straight lines for, for certain input ranges in your functions, then it can still be convex, but for a strictly convex function, there cannot be straight lines, right? It should always be curving upwards. 

Now, Jensen's inequality tells us that expectation of F of X, where X is some random variable, is always greater than or equal to F of expectation of X where F is convex. 

![[Pasted image 20240814183754.png]]

So this is Jensen… so Jensen's inequality tells that the expectation of F of X, where F is a convex function, X is a random variable, and the expectation is taken with respect to the randomness in X, will always be greater than equal to F of the expectation of X, right?

Moreover, if F is strictly convex, strictly convex, then expectation of F of X equals F… if, then, then if expectation F of X equals F of E of X, then X equals expectation of X with probability 1.

There's a lot of jargon here. We will dissect it in a moment. So to, to kind of understand this more intuitively, now this, this picture can help, right?

Let this be some function, F of X, and this is basically X. Use a different color here. Let us also assume that X has a probability distribution associated with it, right? So the green dotted line represents the probability density of, of the random variable X. F is some function of X. 

Now, expectation of X or E would be somewhere here. So let us call this expectation of X, right? So expectation of X is, is, um… so think of it as like the mu if this is a Gaussian, right? That's the expectation of the random variable, right? 

Now, F of expectation of X… so in, in the case where X… let us assume X takes only two possible values. So let us assume…  let us draw another picture here. So this is X. This is F of X. 

Let's assume X takes only two possible values. Let's assume it's a discrete distribution. Here we, we… here X was continuous, but to understand Jensen's inequality, let's assume a discrete setting where X takes only two possible values: this value and this value. May be they are, you know,  0, 1, 2, 3, 4… 1 and 10. Let us assume X takes any one of these values. 

The mean of X… if it takes, with probability half, the value 1 and with its probability, half… value 10, then expectation of X will be 5.5. 

So expectation of X equals 5.5… is 5.5… yeah, 5.5. This over here is F of expectation of X, right? Does that make sense?

This is the expectation of X. You evaluate F at expectation of X, and you get F of expectation of X. That is the right-hand side, right? 

Similarly, with probability half, F of X can takes this value. With probability half, another half, F of X takes this value, right? So this is… let us call this A and B.

So this is F of A, and this is F of B, right? Expectation of F of N… F of B is basically the midpoint between F of A and F of B, right? Because with probability half, it takes this value. With probability half, it takes this value. So the expectation of F of X is this one, right? This is expectation of F of X, right?

It, it so happens that this point will always be the midpoint of… will always be the midpoint of the chord connecting F of A and F of B, right? What Jensen's inequality is telling us is that this point, the point that, that's the, the midpoint of the chord connecting two points on F will always be higher than this point, right? 
![[Pasted image 20240814184628.png]]

So, F of expectation of X is always less than the expectation of F of X, right? Is this clear? Can you raise your hand if you understood this? 

Some of you have not. Okay, can anybody tell me what, what's, what's still confusing here? I can just go over it again.

So F is a convex function which is kind of bending upwards, right? The X axis is, is denotes some random variable. In this case, just for the purpose of understanding Jensen's inequality, let's assume X is, is a discrete… takes on two values, one of two values – either 1 or 10 with equal probability. So the expectation of X is therefore 5.5. That's, that's over here. 

Now, F of 1… you know, let's call it A. You know, F of A is, is this point. So this is F of A. So the height from the X axis to this point, this is F of A. Similarly, if this is B, this is F of B. The expectation of F of X is therefore the midpoint connecting these two points, right? That comes over here. The X… F of expectation of X, where expectation of X was, was 5.5, is this point that comes over here. Jensen's inequality is, is therefore essentially saying that the chord connecting any two points of a convex function always lies above the chord itself, right? 

The expectation… expectation of F of X is higher than F of expectation of X.

Right? Kind of understood?

All right. 

Okay, let's, let's move on. It also tells us that if F is strictly convex, right, F is strictly convex, then F of X equals… if it is… F is strictly convex and if expectation of F of X equals F of expectation of X, then X equals the expectation of X itself, which means X is essentially a constant. 

What does that mean? 

So here is an example of F of X that is strictly convex. 

Now, if expectation of F of X equals F of expectation of X, when can that be possible? Expectation of F of X equals F of expectation of X… when can… let's assume, you know, A and B are here, right? 

This is expectation of F of X. Let us say this is F of expectation of X. 

If F is strictly convex, and if the two are equal, F of X equals expectation of X, then the only way that is possible… that, you know, F of X equals expectation of F… F of X is if X has a probability density… right now, over here, in this dotted line, essentially I am drawing the probability density of X, which is like a Dirac delta function where it has all its mass concentrated at just one point, right? 

In this case, this is expectation of X, and F of expectation of X is here. Also because X always takes on this value with probability 1, F of X also always takes on this prob… this value with probability 1, and therefore F of X equals F of expectation of X equals expectation of F of X, because essentially, you know, the equivalent of the chord connecting two points has length 0 here, right? All the values of F of X are always here, and X always evaluates to the same value next. 

Question. So what's the expectation of a continuous random variable?

If X is a continuous random variable and it has a PDF, let's call it, um, small p of X, right? Then expectation of X is equal to the integral of X times p of X dx. So p in this case is some probability. So the green line… the green dotted line is p of X here.

My question is what is E? So, oh, what's E of F of X? So E of F of X… so if E of F of X is equal to the integral of F of X, p of X dx. Good question.

Right, so this is, this is Jensen's inequality. The reason why we require F to be strictly convex is because if F were not strictly convex, then you could have a case where X… X is… F of X is flat in some place, and X has this density and, expectation of F of X would be here, and F of expectation would be here. Also, F… expectation of F of X would also be the average F of X in this region, which is also a constant. So expectation of F of X would be equal to F of expectation of X, even though X is not constant, because F has a flat region somewhere.

This question. So the question is, uh, in, in the convex case, can we assume that… so, for, for, I mean for this case… yeah. So if, if, for this to hold without X being a constant for X2 and F of X to be the same, then all of X should be distributed in a region where F is flat. 

Yeah, that's right. 

So what are some examples of convex functions? Anybody… example for convex function?

Y equals X square. 

Example for concave function?

Minus X square. Y equals minus X square. 

Yeah, yeah, that's good.

So, examples of convex, concave, and strict: yes or no. Right? So convex function we saw X square is convex. Minus X square is therefore concave. Is this strictly convex and therefore also strictly concave? 

Right? 

Now, another function, Y equals, or F of X equals, equals, say, mx plus C: straight line, is convex. By definition it is convex. A straight line is convex and it is also concave. 

But is it strict? 

No. 

Right. It's fine. Great. Now, what about e to the X? 

Yes, convex. 

Minus e to the X is therefore concave. Is it strict? 

What about log X? So log X is concave and therefore minus log X is convex, right? And it is strict. 

Cool. 

Now how does this… how is this useful for expectation maximization?

Yes.

Question. 

Yeah. A straight line is always convex and concave because, is, is… 

So F double prime is equal to 0 and the definition of convex is that F double prime should be greater than or equal to 0. So, and it is equal to 0, so it satisfies greater than or equal to 0. Similarly for concave, it is less than or equal to 0, and it's equal to 0, so it satisfies less than or equal to 0.

So now… so using Jensen's inequality and with these observations that expectation of F of X is greater than equal to F of expectation of X, and, and so on… we can adapt Jensen's inequality to the concave case where basically it says, if F is concave, right? Example: F of X equals log X, right? 

Then the inequality will switch. So the expectation of log X, instead of greater than equal to, will be less than or equal to log expectation of X. This is also Jensen's inequality.

So now let us derive the EM algorithm, right? So in the EM algorithm, our goal is to maximize log p of Xi comma, theta, where by theta I mean, you know, all the parameters i equals 1 to n. 

You want to maximize this. That is the goal, what we are trying to achieve. This is our end goal: we want to maximize log p of X. However, maximizing log p of X can be hard because the Z's are unobserved. If these were observed, this was very easy, but Z's are unobserved, so it's hard. That's the case where… that's the setting we are in.

Now for the derivation, I'm going to assume one example. So I'm just going to write it as log p of X comma theta, and I'm going to leave the summation out, but basically the whole, the entire derivation that we are going to do, you can include the summation and everything will hold, right? This is just to simplify notation.

So log p of X… we want to maximize this. That's our goal, right? So the first thing we are going to do is write log p of X comma theta is equal to log of the sum over Z p of X Z theta, right?

First we are going to marginalize out Z. Then once we do that, we will define an, you know, some arbitrary probability distribution called q over Z, and write this as log sum over Z q of Z times p of X comma Z theta divided by q of C, where q of Z is greater than 0 for all C.

Some arbitrary probability distribution over Z. It could be anything whatsoever. Any kind of probability distribution over Z such that q of C is greater than 0 everywhere. 

This question. So, so the question is, it's, you know, why is this a hard problem? So it is hard because we are having a summation over here, right? In general, when, in the cases where Z is continuous, this will be an integral. That integral can be, you know, arbitrarily complex. It can be computationally expensive. It can be analytically not possible in cases when we want an analytical solution. 

Good question.

So we come up… you know, q can be any distribution whatsoever as long as q of Z is greater than 0 for all Z. Now we can see that this can be written as log expect… 

Right, what did I do here? 

Nothing basically. So this is the definition of expectation, right? So this is a function of, you know, Z. Some function of Z, right? And think of this as the probability, and this is some function. Therefore, this is just the expectation. Is this clear?

Yeah. 

Okay, so this is, this… just, you know, the expectation. Now we make use of Jensen's inequality, and note that log of the expectation of something is greater than the expectation of the log of the same thing. 

So this is going to be greater than or equal to expectation of Z q log p of X comma Z theta over q of C.

This question. 

F here is log, and log is concave. Log is concave, right? This is our random variable X.

Any, any, any questions on how we apply… how we went from here to here? This is probably the most crucial step, right?

All good? 

Okay.

This… what we see here we will call this, you know, give it a name. We will call it ELBO – evidence lower bound. 

So it is the ELBO of q. And LB of X q theta. 

Jensen's inequality tells us that the ELBO, you know, which is defined to be this term, is always less than or equal to our objective that we want to maximize. Which means now if we find thetas and q's such that we are maximizing the ELBO, then implicitly, for the same values of theta, log p of X is also going up. Does it make sense?

ELBO is defined to be… is… by Jensen's inequality, ELBO is always a lower mound for log p of X. Our likelihood… both of them have theta in them, right? Now if we find values of theta such that we are maximizing the ELBO, then it necessarily means that log p of X at that value of theta is higher and Jensen's inequality gives us that inequality. 

Yes. 

Question.

All right. 

So this is the, the ELBO. This term, ELBO, is, something you will very commonly encounter if you're reading, reading research papers about, you know, generative models or deep generative models. This is, this is a widely used term. ELBO means, you know, the, the lower, the lesser side of the Jensen's inequality of log, log p of X, right?

Our goal is to now… well before we go into our goal, let us make a few more observations. Now, log p of X is greater than equal to ELBO at all times. That's what Jensen's inequality says. But are there cases when log p of X is exactly equal to the ELBO? 

Are there cases… are there cases when log p of X theta is equal to ELBO of X, q, and theta? 

![[Pasted image 20240814194244.png]]

The answer is yes, it is yes. Because of the second part of Jensen's inequality that we saw, right? So Jensen's inequality, we also saw that if F is strictly convex… log X is strictly convex, right? If F is strictly convex, then expectation of F of X… expectation of F of X equals F of expectation of X if and only if X is a constant, right?

So this is one side of Jensen's inequality. This is the other side of Jensen's inequality. The two will be equal if and only if the term inside is a constant. That's what Jensen's inequality told us. Because log is strictly concave… this question.

In this case, we want… we want this entire term over here to be a constant. Now are there cases… so the next question is, are there, you know, under what circumstances is this entire term over here always a constant? 

That's, that's, that's what we're going to answer next. 

It has to be, it has to be independent of Z, so it is constant with respect to Z. Right? So the question now is, in order to make this inequality an equality, because, because log is, because log is strictly concave, the inequality becomes an equality if and only if p of X comma Z theta over q of Z equals some constant C, right? 

Now this implies that… take q of Z over there… q of Z is equal to 1 over C times p of X comma Z, right? We also know that, because this is just a proportionality constant, we can write this as q of Z is proportional to p of X comma Z.

So q of Z is just proportional to p of X comma Z, and in order to make this equal to, we just use the… calculate the proportional normalizing constant that you are summing over Z p of X comma Z theta, right? This is basically p of X comma Z theta divided by… when you marginalize out Z, you just get p of X theta. This is equal to p of Z given X theta.

So the question is, why did we normalize it with p of X?

So because q of Z is proportional to this… and in order to… and we know that this is a probability distribution which means it has to sum up to 1. So this must… the normalizing constant necessary must necessarily be the, um, the sum of this. So the denominator is this term that is summed over all possible values of C. But if something, if you're summing a distribution to 1, is it multivariate? And it helps you to sum over X and Z?

So this is a distribution over Z. p of X Z could be anything. X could be continuous. q of Z is a distribution over Z that must sum up to 1, right? That's proportional to p of X, X comma Z, and the, the… it is proportional, and the corresponding normalizing constant must necessarily be the, um, the sum over the numerators for all possible values of Z because it must, it must sum up to 1.

So when q of Z equals p of Z given X, then Jensen's inequality will change into an equality, right? Lots of moving parts.

Yes. 

So we started with, we started with log p of X, right? Wrote it out as, as, you know, the sum over Z of the joint. You know, there is nothing fancy going on here. We are just marginalizing out the Z, and then we multiply and divide it by some arbitrary distribution q. By multiplying it and dividing it, this… the numerator allowed us to write it in the form of an expectation. The… and this stayed in the denominator. 

Once we wrote it as an expectation, we had log and the expectation, right? So initially we started with the log likelihood, and therefore we use the concave version of Jensen's inequality. 

So this log and this expectation are then swapped, right? We get a greater than equal to expectation of log, right? Once we get this, you know, this is basically Jensen's inequality, the one side and the other side of Jensen's inequality. We call the lower… the lower end of the Jensen's inequality, we just called it ELBO. We just gave it a name, right?

Then we use the, the corollary of Jensen's inequality to look for conditions when this inequality will be exactly equal to. Because log, because log is strictly convex, the corollary of Jensen's inequality gives us a condition when this inequality becomes an equality. The condition is that this term must be a constant. 

Then in order for this to be a constant with respect to Z, it is necessarily the case that q of Z must be equal to the posterior distribution of Z given X. 

Whenever q equals Z given X at that value of theta, then Jensen's inequality becomes a strict… becomes an equality, right?

Now given these two… yes, questions?

So why is… uh… yeah, this… 

All right. 

So we want to find the condition that p of X comma Z divided by q… this is equal to some constant, right? Now let it be equal to some unknown constant C, right? So you can take q of Z over here, right? 

Once you take it over here, this becomes proportional because of some constant times, right? So now what we… we also know that sum over q of Z equals 1, right? Which means the sum over all of, all of the right-hand, uh, terms must be equal to C itself, right? We divide it by C, and we get exactly… 

Okay, so q of Z is equal to the posterior of Z given X at that value of theta.

Thank you. Thanks for asking. 

So based on this, we, we come… we write the EM algorithm, or the more general form of the EM algorithm. We call it the more general form because throughout this, throughout this setting, we have not assumed any specific form for p of X and Z, right? It could be the mixture of Gaussians. It could be any algorithm. This derivation holds for any such latent variable model. 

## The EM Algorithm

All right. So that gives us the algorithm. So the EM algorithm… the EM algorithm basically tells… for… so we have the E-step.

For each i, set qi of Zi is equal to p of Zi given Xi comma theta.

And the M-step. 

Set theta equal to argmax theta of i equal to 1 to n alpha of Xi, qi comma theta. 

Right, so what did we do? We basically, you basically get this EM algorithm where the corresponding E-step is to set q to be the posterior distribution of p of Z given X. In the M-step, we set theta to be equal to the argmax of the ELBO. 

![[Pasted image 20240814201223.png]]

Now why will this, why will this work? To see why this works, let's see this diagram. Now let's suppose this is theta… it's not X. This is theta. This is is arm… so let us assume this is our… so this is log p of X theta, right? 

As we vary theta, log p of X gives us different values. This is not the density. This is the likelihood because the X axis is theta and not X. It's a dotted line because we don't know it, right? It is hard to calculate. We have to marginalize out X, which may be in an integral. 

What we got from this ELBO is that, for any given value of q, the ELBO of, of X for q and theta will always be less than or equal to log p of X. That's what Jensen's inequality gave us. 

So Jensen's inequality tells us that, you know, this is… I'm going to write this as… right, this is one possible ELBO: mix comma q and theta. Let's consider another ELBO. 

So for example, this is another ELBO: X, q, theta.

So for different choices of q, we get different lower bounds of p of X. What Jensen's inequality tells us is that, or the corollary tells us, is that for a given value of theta… let's, let's assume this is our randomly… randomly initialize theta of 0. Right? For this value of theta, if we choose q to be p of Z given X theta 0… for this choice of q, let us call it q0. 

For this choice, if supposing this is the ELBO for q0, then it basically tells us that the ELBO value equals log p of X at this value of k, which means the ELBO touches log p of X at this value of, of theta naught. 

When we are at theta naught, if we choose the q… if we choose to construct an ELBO using q as the posterior distribution with that parameter value, then the ELBO will be tight with respect to the invisible goal that we are trying to maximize at that value of theta. Now if we maximize in the M-step… if we choose a new theta such that we maximize… 

If this is theta 1… that is the, the theta for the next iteration… and now we construct yet another ELBO.

Now the new ELBO will be cons… constructed using, you know, this q… q1, which uses state q1. Then this ELBO will be, will be tight at log p of X at theta 1. 

Depending on the, the choice of the q to be equal to the posterior using the corresponding theta value, the ELBO that we'll get will be tight. I… will be touching the, the log p of X. That's, that's what Jensen's inequality is… corollary tells us. 

Now if we started theta naught, maximize the, the ELBO, that is the M-step… we get a new theta 1, construct a new ELBO and maximize that one, and we reach here. So this will be theta 2, and here we construct yet another ELBO such that it is tight at, at this value of, of theta 2, and so on. 

If we repeat this over and over by constructing different lower bounds at each value of theta, and we are maximizing the lower bound, and so on, then we will eventually reach a local optima where the algorithm converges, which means, you know, theta stops changing, and that's essentially what the EM algorithm does. 

So this is the rough intuition of, or, or the, or the visual intuition of how the EM algorithm works. In the next class, we will, we will go through a proof to show that it actually converges, you know, more than just, you know, just simply drawing some pictures. 

Yes, questions?

Yes. 

Exactly. So how do we compute the ELBO? The ELBO is, is exactly this… 

Exactly.

So in the next class, we'll see an example of how we apply to Gaussian Mixture Models where it might be more clear. But for now, for, you know, for the purpose of this lecture, it's enough to have this abstract view of how the EM algorithm works in general. In the next class, we'll apply it to Gaussian Mixture Models and see exactly how each of the steps work. 

Is there any other question? 

Yes. 

Question is theta given as Z equals theta transpose X.

Now theta… theta here is some unknown parameter of the model.

No, no, that we don't make any linearity assumptions you… ...don't make any assumptions on on the form of p of X given Z. It could be any arbitrarily complex function, it could be a neural network or something. You know, this framework is very general and that's the beauty of EM algorithm. So this derivation of EM algorithm that we saw here, it makes no assumption whatsoever on the form of p of X comma Z or any of the individual distributions. We just require them to be, you know, valid probability distributions, but they could be arbitrarily complex and this will still hold. Yeah, good question.

Okay, I think we'll stop here. In the next class, we'll start from here and we'll look at an example of how EM can be used to solve the Gaussian Mixture Model, and then look at the proof of convergence of EM algorithm.

## Important Topics and Keywords:

**Unsupervised Learning:**

* **Definition:** Learning from data without labeled examples or "supervision."
* **Goal:** Discover hidden structures, patterns, and insights within the data.
* **Example Applications:** Clustering (e.g., customer segmentation), density estimation (e.g., anomaly detection).

**K-means Clustering:**

* **Goal:** Partition data points into k clusters.
* **Algorithm:** Iteratively assigns points to nearest centroid, then updates centroids based on assigned points.
* **Distortion Function:** Objective function minimized by k-means.
* **Convergence:** Guaranteed to converge to a local minimum of the distortion function.
* **Non-Convexity:** Different initializations can lead to different final clusterings.

**Density Estimation:**

* **Goal:** Estimate the underlying probability density function from which data points are sampled.
* **Challenge:** Finding a smooth function to represent discrete observations.
* **Gaussian Mixture Model (GMM):** A common approach using a weighted sum of Gaussian distributions to approximate the density.

**Gaussian Mixture Model (GMM):**

* **Assumption:** Data points are generated from a mixture of k Gaussian distributions.
* **Parameters:** Mixing coefficients (phi), means (mu), and covariances (sigma) of the Gaussian components.
* **Latent Variables:** Cluster assignments (Zi) are unobserved.
* **Maximum Likelihood Estimation (MLE):** Used to estimate model parameters.
* **Soft Clustering:** Each point has a probability of belonging to each cluster.

**Expectation Maximization (EM) Algorithm:**

* **Goal:** Perform MLE in the presence of latent variables.
* **E-step:** Estimate posterior probabilities of latent variables given observed data and current parameter estimates.
* **M-step:** Update parameter estimates to maximize the expected log-likelihood (ELBO).
* **ELBO (Evidence Lower Bound):** A lower bound on the log-likelihood.
* **Jensen's Inequality:** Used to derive the EM algorithm and guarantee convergence.
* **Convergence:**  Iteratively maximizes a lower bound on the log-likelihood, eventually reaching a local maximum.
* **Generality:**  Applicable to a wide range of latent variable models, including GMMs.

**Other Key Terms:**

* Supervised Learning
* Reinforcement Learning
* Coordinate Descent
* Convex Function
* Concave Function
* Latent Variable
* Prior Probability
* Posterior Probability
* Evidence
* Soft k-means
* Jensen's Inequality
* ELBO (Evidence Lower Bound)
* Anomaly Detection
* Generative Models 
* Variational Autoencoders
* Generative Adversarial Networks (GANs)
* Flow-Based Models
