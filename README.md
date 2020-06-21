# fAmbientOcclusion
Ambient Occlusion for Universal RP (as same as PPSv2)

This feature should be useful for you who don't wait Ambient Occlusion for official support.  
Shader files is same as that of PPSv2.

![image](https://user-images.githubusercontent.com/24952685/75114171-39eb6500-5697-11ea-83a7-1991033c0f74.png)

## How to use
To set for your project
- Add only 2 files anywhere.
    - fAmbientOcclusion.cs
    - fAmbientOcclusionFeature.cs
- Import PostProcessingStackV2 package.
![image](https://user-images.githubusercontent.com/24952685/75114679-8f297580-569b-11ea-8bda-67670c9ef50f.png)
- Add ScriptableRendererFeature(fAmbientOcclusion Feature) to FowardRendererData.
- Add VolumeComponent(fAmbientOcclusion) to PostProcessVolume.
![image](https://user-images.githubusercontent.com/24952685/75114194-656e4f80-5697-11ea-903e-a559c86da5a5.png)

## Known issue (I won't fix. Please wait for official supporting.)
- No supported XR Rendering
- No supported Deferred Rendering
- No supported ScalableAmbientObscurance Mode

## License
Licensed under the Unity Companion License for Unity-dependent projects--see [Unity Companion License](http://www.unity3d.com/legal/licenses/Unity_Companion_License). 

Unless expressly provided otherwise, the Software under this license is made available strictly on an “AS IS” BASIS WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. Please review the license for details on these and other terms and conditions.
