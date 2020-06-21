using System;
using System.Reflection;
using System.Collections.Generic;

namespace UnityEngine.Rendering.Universal {
	[Serializable, VolumeComponentMenu("Ambient Occulusion from PPSv2")]
	public sealed class fAmbientOcclusion : VolumeComponent, IPostProcessComponent {
		#region DEFINE
		/// <summary>
		/// A volume parameter holding a <see cref="AmbientOcclusionMode"/> value.
		/// </summary>
		[Serializable]
		public sealed class AmbientOcclusionModeParameter : VolumeParameter<AmbientOcclusionMode> { }

		/// <summary>
		/// A volume parameter holding a <see cref="AmbientOcclusionQuality"/> value.
		/// </summary>
		[Serializable]
		public sealed class AmbientOcclusionQualityParameter : VolumeParameter<AmbientOcclusionQuality> { }
		#endregion


		// WIP: ScalableAO
		///// <summary>
		///// The number of sample points, which affects quality and performance. Lowest, Low and Medium
		///// passes are downsampled. High and Ultra are not and should only be used on high-end
		///// hardware.
		///// </summary>
		//[Tooltip("The ambient occlusion method to use. \"Multi Scale Volumetric Obscurance\" is higher quality and faster on desktop & console platforms but requires compute shader support.")]
		//public AmbientOcclusionModeParameter mode = new AmbientOcclusionModeParameter { value = AmbientOcclusionMode.MultiScaleVolumetricObscurance };
		/// <summary>
		/// The strength of the motion blur filter. Acts as a multiplier for velocities.
		/// </summary>
		[Tooltip("The quality of the effect. Lower presets will result in better performance at the expense of visual quality.")]
		public ClampedFloatParameter intensity = new ClampedFloatParameter(0f, 0f, 4f);

		/// <summary>
		/// A custom color to use for the ambient occlusion.
		/// </summary>
		[ColorUsage(false), Tooltip("The custom color to use for the ambient occlusion. The default is black.")]
		public ColorParameter color = new ColorParameter(Color.black);

		// TIPS: Not supported Deferred Rendering
		///// <summary>
		///// Only affects ambient lighting. This mode is only available with the Deferred rendering
		///// path and HDR rendering. Objects rendered with the Forward rendering path won't get any
		///// ambient occlusion.
		///// </summary>
		//[Tooltip("Check this box to mark this Volume as to only affect ambient lighting. This mode is only available with the Deferred rendering path and HDR rendering. Objects rendered with the Forward rendering path won't get any ambient occlusion.")]
		//public BoolParameter ambientOnly = new BoolParameter(true);


		#region MSVO-only parameters
		/// <summary>
		/// The tolerance of the noise filter to changes in the depth pyramid.
		/// </summary>
		public ClampedFloatParameter noiseFilterTolerance = new ClampedFloatParameter(0f, -8f, 0f); // Hidden

		/// <summary>
		/// The tolerance of the bilateral blur filter to depth changes.
		/// </summary>
		public ClampedFloatParameter blurTolerance = new ClampedFloatParameter(-4.6f, -8f, -1f); // Hidden

		/// <summary>
		/// The tolerance of the upsampling pass to depth changes.
		/// </summary>
		public ClampedFloatParameter upsampleTolerance = new ClampedFloatParameter(-12f, -12f, -1f); // Hidden

		/// <summary>
		/// Modifies the thickness of occluders. This increases dark areas but also introduces dark
		/// halo around objects.
		/// </summary>
		[Tooltip("This modifies the thickness of occluders. It increases the size of dark areas and also introduces a dark halo around objects.")]
		public ClampedFloatParameter thicknessModifier = new ClampedFloatParameter(1f, 1f, 10f);
		#endregion


		// WIP: ScalableAO
		#region SAO-only parameters
		///// <summary>
		///// Radius of sample points, which affects extent of darkened areas.
		///// </summary>
		//[Tooltip("The radius of sample points. This affects the size of darkened areas.")]
		//public FloatParameter radius = new FloatParameter(0.25f);

		///// <summary>
		///// The number of sample points, which affects quality and performance. Lowest, Low and Medium
		///// passes are downsampled. High and Ultra are not and should only be used on high-end
		///// hardware.
		///// </summary>
		//[Tooltip("The number of sample points. This affects both quality and performance. For \"Lowest\", \"Low\", and \"Medium\", passes are downsampled. For \"High\" and \"Ultra\", they are not and therefore you should only \"High\" and \"Ultra\" on high-end hardware.")]
		//public AmbientOcclusionQualityParameter quality = new AmbientOcclusionQualityParameter { value = AmbientOcclusionQuality.Medium };
		#endregion


		/// <summary>
		/// Is the component active?
		/// </summary>
		/// <returns>True is the component is active</returns>
		public bool IsActive() => intensity.value > 0f && this.active;

		/// <summary>
		/// Is the component compatible with on tile rendering
		/// </summary>
		/// <returns>false</returns>
		public bool IsTileCompatible() => false;

#if UNITY_EDITOR
		protected override void OnEnable() {
			base.OnEnable();

			var forward = UniversalRenderPipeline.asset.scriptableRenderer as ForwardRenderer;
			if (forward != null) {
				var prop = typeof(ScriptableRenderer).GetProperty("rendererFeatures", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.GetProperty);
				var features = prop.GetValue(forward) as List<ScriptableRendererFeature>;
				bool missing = true;
				foreach (var f in features) {
					if (f is fAmbientOcclusionFeature) {
						missing = false;
						break;
					}
				}

				// ScriptableRendererFeature is Missing when you reimport the project...
				if (missing)
					Debug.LogWarning("Missing fAO Feature...");
			}
		}
#endif
	}
}
