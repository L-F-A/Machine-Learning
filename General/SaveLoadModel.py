import pickle


#Saving or loading a particular model. Meaning saving or loading an object containing 
#the different parameters of one ML model
#
#If saving, should looks like:
#object = Object()  
#filehandler = open(filename, 'w')  
#pickle.dump(object, filehandler)
#
#If loading, should looks like:
#filehandler = open(filename, 'r')  
#object = pickle.load(filehandler)


def SaveLoadModel(filename,action='s',obj=None):
	
	if action is 's':
		object_to_save = obj  
		file_obj = open(filename, 'w')  
		pickle.dump(object_to_save, file_obj)
		return 0.
	elif action is 'L':
		filehandler = open(filename,'r')
		return pickle.load(filehandler)
	

