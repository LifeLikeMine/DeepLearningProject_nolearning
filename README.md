# 딥러닝 프로젝트
> 딥러닝 프로젝트 NoLearning팀

**딥러닝 프로젝트 주제 - Road following + Collision Avoidance을 활용한 과속방지턱**


시연 영상 X

## 개발 환경

<img src="https://img.shields.io/badge/#F37626?style=flat&logo=Jupyter&logoColor=white"/>

## 사용 방법

- Road following
https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/road_following
예시대로 dataset 추출 및 모델 생성

- Collision Avoidance
https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/collision_avoidance
예시대로 dataset 추출 및 모델 생성
dataset중 block에 충돌상황 사진 촬영

1. 각각 모델 생성
2. 프로젝트 파일 실행


## 코드블럭 설명

```python


def execute(change):
    global angle, angle_last, blocked_slider, robot, count_stops, stop_time, go_on, x, y, blocked_threshold
    global speed_value, steer_gain, steer_dgain, steer_bias
                
    steer_gain = steering_gain_slider.value
    steer_dgain = steering_dgain_slider.value
    steer_bias = steering_bias_slider.value
       
    image_preproc = preprocess(change['new']).to(device)
     
    #Collision Avoidance 모델:
    
    prob_blocked = float(F.softmax(model_trt_collision(image_preproc), dim=1).flatten()[0])
    
    blocked_slider.value = prob_blocked    
    stop_time=stopduration_slider.value
    
    if go_on == 1:    
        if prob_blocked > blocked_threshold.value: 
            count_stops += 1
            go_on = 2
        else:
            #road following 감지
            go_on = 1
            count_stops = 0
            xy = model_trt(image_preproc).detach().float().cpu().numpy().flatten()        
            x = xy[0]            
            y = (0.5 - xy[1]) / 2.0
            speed_value = speed_control_slider.value
    else:
        count_stops += 1
        if count_stops < stop_time:
            x = 0.0 
            y = 0.0
            speed_value = speed_control_slider.value
            speed_value = speed_value/3 # 과속 방지턱 감지시 속도 감소
        else:
            go_on = 1
            count_stops = 0
            
    
```



## 참고

1. Road Following: [https://github.com/kyechan99/capsule-render](https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/road_following)
2. Collision Avoidance: [https://yermi.tistory.com/entry/%EA%BF%80%ED%8C%81-Github-Readme-%EC%98%88%EC%81%98%EA%B2%8C-%EA%BE%B8%EB%AF%B8%EA%B8%B0-Readme-Header-Badge-Widget-%EB%93%B1
](https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/collision_avoidance)
3. https://developer.nvidia.com/embedded/community/jetson-projects
