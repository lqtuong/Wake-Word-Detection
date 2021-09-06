import os
from threading import Thread
from time import sleep
from time import gmtime, strftime
STATUS_PLAY = 1
STATUS_PAUSE = 0

class Mplayer:
	status = 0
	source = ''
	pipe = ''
	def __init__(self):
		print("MPLAYER init")
		self.pipe = '/tmp/mplayer'+  strftime("%Y%m%d%H%M%S", gmtime())
		os.system('mkfifo ' + self.pipe)

	def set_source(self, source):
		self.source = source
		os.system('echo loadfile  ' + source + ' > '+ self.pipe + ' ')
		self.status = STATUS_PLAY

	def play(self):
		os.system('mplayer -slave -idle -input file=' + self.pipe + ' > /dev/null 2>&1')
#		if self.status:
#			os.system('echo pause > ' + self.pipe)
#			self.status = STATUS_PAUSE
#		else:
#			print("Mplayer is playing")

	def pause(self):
		if self.status == STATUS_PAUSE:
			os.system('echo pause > ' + self.pipe)
			self.status = STATUS_PLAY
		else:
			print("Mplayer is pausing")

	def set_volume(self, value):
		os.system('echo volume ' + value + ' > ' + self.pipe)

	def quit(self):
		os.system('echo quit > ' + self.pipe)
		os.system('rm ' + self.pipe)
