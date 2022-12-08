# 딥러닝 프로젝트
> 딥러닝 프로젝트 NoLearning팀

**딥러닝 프로젝트 주제 - Road following + Collision Avoidance을 활용한 과속방지턱**

카메라 오류로 인해 구현 실패

시연 영상 X

## 필요성

- 자율주행시 도로에 존재하는 과속방지턱 상황을 대처할수있음

## 개발 환경

<img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=Jupyter&logoColor=white"/> <img src="https://img.shields.io/badge/NVIDIA-76B900?style=flat&logo=NVIDIA&logoColor=white"/>

## 사용 방법

- Road following
https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/road_following
예시대로 dataset 추출 및 모델 생성

- Collision Avoidance
https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/collision_avoidance
예시대로 dataset 추출 및 모델 생성
dataset중 block에 충돌상황 사진 촬영

1. 각각 모델 생성
3. 프로젝트 파일 실행


## 코드블럭 설명

- 슬라이더 계산

```python

#Road Following sliders
speed_control_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, description='speed control')
steering_gain_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.04, description='steering gain')
steering_dgain_slider = ipywidgets.FloatSlider(min=0.0, max=0.5, step=0.001, value=0.0, description='steering kd')
steering_bias_slider = ipywidgets.FloatSlider(min=-0.3, max=0.3, step=0.01, value=0.0, description='steering bias')

display(speed_control_slider, steering_gain_slider, steering_dgain_slider, steering_bias_slider)

#Collision Avoidance sliders
blocked_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, orientation='horizontal', description='blocked')
stopduration_slider= ipywidgets.IntSlider(min=1, max=1000, step=1, value=10, description='time for stop') 
blocked_threshold= ipywidgets.FloatSlider(min=0, max=1.0, step=0.01, value=0.8, description='blocked threshold')

display(image_widget)

display(ipywidgets.HBox([blocked_slider, blocked_threshold, stopduration_slider]))

```

- Road following + Collision Avoidance

```python


def execute(change):
    global angle, angle_last, blocked_slider, robot, count_slow, slow_time, go_on, x, y, blocked_threshold
    global speed_value, steer_gain, steer_dgain, steer_bias
                
    steer_gain = steering_gain_slider.value
    steer_dgain = steering_dgain_slider.value
    steer_bias = steering_bias_slider.value
       
    image_preproc = preprocess(change['new']).to(device)
     
    #Collision Avoidance 모델:
    
    prob_blocked = float(F.softmax(model_trt_collision(image_preproc), dim=1).flatten()[0])
    
    blocked_slider.value = prob_blocked    
    slow_time=slowduration_slider.value
    
    if go_on == 1:    
        if prob_blocked > blocked_threshold.value: 
            count_slow += 1
            go_on = 2
        else:
            #road following 감지
            go_on = 1
            count_slow = 0
            xy = model_trt(image_preproc).detach().float().cpu().numpy().flatten()        
            x = xy[0]            
            y = (0.5 - xy[1]) / 2.0
            speed_value = speed_control_slider.value
    else:
        count_slow += 1
        if count_slow < slow_time:
            x = 0.0 
            y = 0.0
            speed_value = speed_control_slider.value
            speed_value = speed_value/3 # 과속 방지턱 감지시 속도 감소
        else:
            go_on = 1
            count_slow = 0
            
    angle = math.atan2(x, y)        
    pid = angle * steer_gain + (angle - angle_last) * steer_dgain
    steer_val = pid + steer_bias 
    angle_last = angle
    robot.left_motor.value = max(min(speed_value + steer_val, 1.0), 0.0)
    robot.right_motor.value = max(min(speed_value - steer_val, 1.0), 0.0)          
            
    
```



## 참고

1. Road Following: https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/road_following
2. Collision Avoidance: https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/collision_avoidance
3. https://developer.nvidia.com/embedded/community/jetson-projects
