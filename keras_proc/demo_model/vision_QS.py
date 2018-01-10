#再针对一幅图片使用自然语言提问时，该模型能提供关于该图片的一个单词的答案

"""
这个模型将NLP的问题和图片分别映射为特征向量，将两者合并后训练一个logstic回归层
从一系列的可能的回答中挑选一个
"""
from keras.layers import Conv2D,MaxPooling2D,Flatten,concatenate
from keras.layers import Input,LSTM,Embedding,Dense
from keras.models import Model,Sequential
#First ,define a vision model using a Seauential model
#this model will encode an image into a vector

vision_model=Sequential()
vision_model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(224,224,3)))
vision_model.add(Conv2D(64,(3,3),activation='relu'))
vision_model.add(MaxPooling2D((2,2)))
vision_model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
vision_model.add(Conv2D(128,(3,3),activation='relu'))
vision_model.add(MaxPooling2D((2,2)))
vision_model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
vision_model.add(Conv2D(256,(3,3),activation='relu'))
vision_model.add(Conv2D(256,(3,3),activation='relu'))
vision_model.add(MaxPooling2D((2,2)))
vision_model.add(Flatten())

#now get a tensor with the output of our vision model:
image_input=Input(shape=(224,224,3))
encode_image=vision_model(image_input)

"""
Next ,define a NLP model to encode the question into a vector
Each question will be at most 100 word long
and we will index words as integers from 1 to 9999
"""
question_input=Input(shape=(100,),dtype='int32')
embedded_question=Embedding(input_dim=10000,output_dim=256,input_length=100)(question_input)
encode_question=LSTM(256)(embedded_question)

#let's concatenate the question vector and the image vector:
merged=concatenate([encode_image,encode_question])

#And let's train a logstic regression over 1000 words on top:
output_result=Dense(1000,activation='softmax')(merged)

#Final model
vqa_model=Model(input=[image_input,question_input],output=output_result)

print("Ready to Go")
"""
为模型提供一个短视频（100）帧然后向模型提问一个关于该视频的问题
"""
from keras.layers import TimeDistributed

video_input=Input((100,224,224,3))
# this is video encode
encode_frame_sequence=TimeDistributed(vision_model)(video_input)
encode_video=LSTM(256)(encode_frame_sequence)

# this is a model-level resprenation 
question_encoder=Model(input=question_input,outputs=encode_question)

#encode the question
video_question_input=Input(shape=(100,),dtype='int32')
encode_video_question=question_encoder(video_question_input)

#this is our video question answering model:
merged_VS=concatenate([encode_video,encode_video_question])
output_VS=Dense(1000,activation='softmax')(merged_VS)
video_qa_model=Model(input=[video_input,video_question_input],outputs=output_VS)
print("Ready to GO 2!")