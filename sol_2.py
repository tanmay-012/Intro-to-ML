import numpy as np

global dt_words

def my_fit( words, verbose = False ):
	model = Tree( min_leaf_size = 1, max_depth = 15 )
	model.fit( words, verbose )
	return model


class Tree:
	def __init__( self, min_leaf_size, max_depth ):
		self.root = None
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
	
	def fit( self, words, verbose = False ):
		self.root = Node( depth = 0, parent = None )
		if verbose:
			print( "root" )
			print( "└───", end = '' )
		# The root is trained with all the words
		self.root.fit( words, my_words_idx = np.arange(len(words)), min_leaf_size = self.min_leaf_size, max_depth = self.max_depth, verbose = verbose )


class Node:
	def __init__( self, depth, parent ):
		self.depth = depth
		self.parent = parent
		self.my_words_idx = None
		self.children = {}
		self.is_leaf = True
		self.query_idx = None
	
	def get_query( self ):
		return self.query_idx
	
	def get_child( self, response ):
		if self.is_leaf:
			child = self
		else:
			if response not in self.children:
				print( f"Unknown response {response} -- need to fix the model" )
				response = list(self.children.keys())[0]
			
			child = self.children[ response ]
			
		return child
	
	def process_leaf( self, my_words_idx):
		return my_words_idx[0]
	
	def reveal( self, word, query ):
		mask = [ *( '_' * len( word ) ) ]
		for i in range( min( len( word ), len( query ) ) ):
			if word[i] == query[i]:
				mask[i] = word[i]
		return ' '.join( mask )

	def get_entropy( self, counts ):
		assert np.min( counts ) > 0, "Elements with zero or negative counts detected"
		num_elements = counts.sum()
		if num_elements <= 1:
			return 0
		proportions = counts / num_elements
		return np.sum( proportions * np.log2( counts ) )

	def single_split( self, words, my_words_idx, query, verbose ):
		split_dict = {}
		count_dict = {}
		for idx in my_words_idx:
			mask = self.reveal( words[ idx ], query )
			if mask not in split_dict:
				split_dict[ mask ] = []
				count_dict[ mask ] = 0
			split_dict[ mask ].append( idx )
			count_dict[ mask ] += 1
		entropy = self.get_entropy( np.array( list( count_dict.values())))
		return (entropy, split_dict)

	def process_node( self, words, my_words_idx, verbose ):
		split_dict = {}
		entropy = np.inf
		query_idx = -1
		query = ""
		if len( my_words_idx ) == len( words ):
			(entropy, split_dict) = self.single_split(words, my_words_idx, query, verbose)
		elif self.depth == 1:
			for i in range(50):
				val = np.random.randint(len(my_words_idx))
				(entropy_cmp, split_dict_cmp) = self.single_split(words, my_words_idx, words[my_words_idx[val]], verbose)
				if entropy_cmp < entropy:
					entropy = entropy_cmp
					query_idx = my_words_idx[val]
					split_dict = split_dict_cmp
		else:
			for i in my_words_idx:
				(entropy_cmp, split_dict_cmp) = self.single_split(words, my_words_idx, words[i], verbose)
				if entropy_cmp < entropy:
					entropy = entropy_cmp
					query_idx = i
					split_dict = split_dict_cmp
		return ( query_idx, split_dict )
	
	def fit( self, words, my_words_idx, min_leaf_size, max_depth, fmt_str = "    ", verbose = False ):
		self.my_words_idx = my_words_idx
		
		if len( my_words_idx ) <= min_leaf_size or self.depth >= max_depth:
			self.is_leaf = True
			self.query_idx = self.process_leaf( self.my_words_idx)
			if verbose:
				print( '█' )
		else:
			self.is_leaf = False
			( self.query_idx, split_dict ) = self.process_node( words, self.my_words_idx, verbose )
			
			if verbose:
				print( dt_words[ self.query_idx ] )
			
			for ( i, ( response, split ) ) in enumerate( split_dict.items() ):
				if verbose:
					if i == len( split_dict ) - 1:
						print( fmt_str + "└───", end = '' )
						fmt_str += "    "
					else:
						print( fmt_str + "├───", end = '' )
						fmt_str += "│   "
				
				# Create a new child for every split
				self.children[ response ] = Node( depth = self.depth + 1, parent = self )
				
				# Recursively train this child node
				self.children[ response ].fit(words, split, min_leaf_size, max_depth, fmt_str, verbose )