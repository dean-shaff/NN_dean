"""
Building on Logistic Regression example from theano tutorial. 
"""
import theano 
import theano.tensor as T
import numpy as np 
from load_mnist import load_data

class MLP(object):

	def __init__(self,rng,dim,activation):
		"""
		dim is a tuple or list containing the dimensions of the 
		neural net: [4,5,2] will have 4 inputs, 5 hidden neurons,
		and 2 outputs 
		L1 and L2 are regularizations to ensure quick convergence 
		"""
		self.W = [] 
		self.b = [] 
		for i in xrange(1,len(dim)):
			if i == len(dim) - 1:				
				self.W.append(theano.shared(
					value=np.zeros(
						(dim[i-1], dim[i]),
						dtype=theano.config.floatX
					),
					name='W',
					borrow=True
				))
				self.b.append(theano.shared(
					value=np.zeros(
						(dim[i],),
						dtype=theano.config.floatX
					),
					name='b',
					borrow=True
				))
			else:
				W_values = np.asarray(
							rng.uniform(
								low=-np.sqrt(6. / (dim[i-1] + dim[i])),
								high=np.sqrt(6. / (dim[i-1] + dim[i])),
								size=(dim[i-1], dim[i])
					),
					dtype=theano.config.floatX
				)
				self.W.append(theano.shared(value=W_values, name='W', borrow=True))

				self.b.append(theano.shared(
					value=np.zeros(
						(dim[i],),
						dtype=theano.config.floatX
					),
					name='b',
					borrow=True
				))
		self.dim = dim 
		self.act = activation

		self.L1 = 0
		self.L2_sqr = 0 
		for i in xrange(len(self.W)):
			self.L1 += abs(self.W[i]).sum()
			self.L2_sqr += (self.W[i]**2).sum()
		# self.L1 = tuple(self.L1)
		# self.L2_sqr = tuple(self.L2_sqr)
		# self.L1 = (
		# 	abs(self.W[0]).sum()
		# 	+ abs(self.W[1::]).sum()
		# )

		# square of L2 norm ; one regularization option is to enforce
		# square of L2 norm to be small
		# self.L2_sqr = (
		# 	(self.hiddenLayer.W ** 2).sum()
		# 	+ (self.logRegressionLayer.W ** 2).sum()
		# )

	def feed_forward(self,x):
		"""
		Calculate the value(s) of the neural net at the output level.
		"""
		y_guess = self.act(T.dot(x,self.W[0])+ self.b[0])
		for i in xrange(1,len(self.W)):
			W = self.W[i]
			b = self.b[i]
			if i == len(self.W)-1:
				y_guess = T.nnet.softmax(T.dot(y_guess,W)+b)
			else: 
				y_guess = self.act(T.dot(y_guess,W)+b)

		return y_guess 

	def loss(self,x,y):
		"""
		Still not sure exactly how this guy works. 
		"""
		y_guess = self.feed_forward(x)
		return -T.mean(T.log(y_guess)[T.arange(y.shape[0]), y])

	def error(self,x,y):

		y_pred = T.argmax(self.feed_forward(x),axis=1)

		return T.sum(T.neq(y_pred,y))


	def SGD(self,training_data,test_data,L1_reg,L2_reg,learning_rate, n_epochs,mini_batch_size):

		n_train = training_data[0].get_value(borrow=True).shape[0] #training data is shared variable
		n_test =  test_data[0].get_value(borrow=True).shape[0]
		train_x, train_y = training_data
		test_x, test_y = test_data 

		n_train_batchs = n_train/mini_batch_size

		index = T.lscalar()
		x = T.matrix('x')
		y = T.ivector('y')

		cost = (
			self.loss(x,y)
			+ L1_reg * self.L1
			+ L2_reg * self.L2_sqr
		)
	

		test_model = theano.function(
			inputs = [],
			outputs = self.error(x,y),
			givens={
				x: test_x,
				y: test_y
			}
		 )
		
		g_W = [T.grad(cost=cost, wrt=self.W[i]) for i in xrange(len(self.W))]
		g_b = [T.grad(cost=cost, wrt=self.b[i]) for i in xrange(len(self.b))]

		updates_W = [(self.W[i],self.W[i] - learning_rate*g_W[i]) for i in xrange(len(self.W))]
		updates_b = [(self.b[i], self.b[i] - learning_rate*g_b[i]) for i in xrange(len(self.b))]
		
		updates = updates_W+updates_b
		train_model = theano.function(
			inputs=[index],
			outputs=cost,
			updates=updates,
			givens={
				x: train_x[index*mini_batch_size: (index+1)*mini_batch_size],
				y: train_y[index*mini_batch_size: (index+1)*mini_batch_size]
			}
		) #like fucking C 

		print("... starting to train")
		error_start = test_model()
		print("Initial error: {}/{}".format(error_start,n_test))
		for i in xrange(n_epochs):
			for index in xrange(n_train_batchs):
				avg_cost = train_model(index)

			# print(np.allclose(self.W.get_value(),W_test))
			
			error_epoch = test_model()
			print("Error for this epoch: {}/{}".format(error_epoch,n_test))


def main():
	# can use sigmoid by inputting T.nnet.sigmoid
	rng = np.random.RandomState(1234)
	training_data, valid_data, test_data = load_data('mnist.pkl.gz')
	trainer = MLP(rng,[28*28,500,10],T.tanh)
	m_b_s = 100 #mini_batch_size
	n_epoch = 1000
	eta = 0.01
	L1_reg = 0.00
	L2_reg = 0.0001
	trainer.SGD(training_data,test_data,L1_reg, L2_reg,eta,n_epoch,m_b_s)

if __name__ == '__main__':
	main()










