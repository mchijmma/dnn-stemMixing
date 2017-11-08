#Data augmentation 
from datetime import datetime
import os
import sox
from fnmatch import fnmatch
import numpy as np

startTime = datetime.now()

# from -4 to 4 semitones, 16 in total. 
semitones = np.concatenate((np.linspace(-4,-0.5,8),np.linspace(0.5,4,8)))


pattern = "*.wav"
counter = 0
counter2 = 0

for path, subdirs, files in os.walk('./Data/MedleyDB/Audio/'):
    
    for name in files:
      
        if fnmatch(name, pattern):
          
            print name
            counter = counter + 1
            #print os.path.join(path, name) 
            for st in semitones:
                counter2 = counter2 + 1
                # create trasnformer
                tfm = sox.Transformer()
                # pitch shift
                tfm.pitch(st)            
                # create the output file.
                name2 = name.split('_')
                name2[1] = name2[1] + '%+d' % (int(100*st))
                name2 = '_'.join(name2)
                
                tfm.build(os.path.join(path, name), os.path.join(path, name2))
                # see the applied effects
                #tfm.effects_log

print '\nExecuted in: %s. \n %d stems/raw were augmented by 8 tracks each, %d in stem/raw in total.' % (str(datetime.now() - startTime), int(counter/3), int((counter2+counter)/3))

