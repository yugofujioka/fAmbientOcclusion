using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;


namespace UnityEngine.Rendering.Universal {
	#region ENUM
	/// <summary>
	/// Ambient occlusion modes.
	/// </summary>
	public enum AmbientOcclusionMode {
		/// <summary>
		/// A standard implementation of ambient obscurance that works on non modern platforms. If
		/// you target a compute-enabled platform we recommend that you use
		/// <see cref="MultiScaleVolumetricObscurance"/> instead.
		/// </summary>
		ScalableAmbientObscurance,

		/// <summary>
		/// A modern version of ambient occlusion heavily optimized for consoles and desktop
		/// platforms.
		/// </summary>
		MultiScaleVolumetricObscurance
	}

	/// <summary>
	/// Quality settings for <see cref="AmbientOcclusionMode.ScalableAmbientObscurance"/>.
	/// </summary>
	public enum AmbientOcclusionQuality {
		/// <summary>
		/// 4 samples + downsampling.
		/// </summary>
		Lowest,

		/// <summary>
		/// 6 samples + downsampling.
		/// </summary>
		Low,

		/// <summary>
		/// 10 samples + downsampling.
		/// </summary>
		Medium,

		/// <summary>
		/// 8 samples.
		/// </summary>
		High,

		/// <summary>
		/// 12 samples.
		/// </summary>
		Ultra
	}
	#endregion

	public class fAmbientOcclusionFeature : ScriptableRendererFeature {
		#region DEFINE
		[System.Serializable]
		public class AOSettings {
			public RenderPassEvent Event = RenderPassEvent.AfterRenderingSkybox;
			public LayerMask LayerMask = 0;
			public Shader shader = null;
			/// <summary>
			/// The compute shader used for the first downsampling pass of MSVO.
			/// </summary>
			public ComputeShader multiScaleAODownsample1;
			/// <summary>
			/// The compute shader used for the second downsampling pass of MSVO.
			/// </summary>
			public ComputeShader multiScaleAODownsample2;
			/// <summary>
			/// The compute shader used for the render pass of MSVO.
			/// </summary>
			public ComputeShader multiScaleAORender;
			/// <summary>
			/// The compute shader used for the upsampling pass of MSVO.
			/// </summary>
			public ComputeShader multiScaleAOUpsample;
		}
		#endregion

		public AOSettings settings = new AOSettings();

		private AOPass aoPass;

		#region AO Pass
		partial class AOPass : ScriptableRenderPass {
			private const string PASS_NAME = "fAO";
			private ProfilingSampler profilingSampler = new ProfilingSampler(PASS_NAME);
			private static readonly int MAIN_TEX = Shader.PropertyToID("_MainTex");
			private static readonly int AO_TARGET = Shader.PropertyToID("AO_Target");

			private fAmbientOcclusion volumeComponent;
			private AOSettings settings = null;
			private Material blitMaterial;
			private Mesh triangle = null;
			readonly int[] m_ScaledWidths = new int[7];
			readonly int[] m_ScaledHeights = new int[7];
			readonly float[] m_InvThicknessTable = new float[12];
			readonly float[] m_SampleWeightTable = new float[12];


			readonly int[] m_SampleCount = { 4, 6, 10, 8, 12 };
			// The arrays below are reused between frames to reduce GC allocation.
			readonly float[] m_SampleThickness =
			{
				Mathf.Sqrt(1f - 0.2f * 0.2f),
				Mathf.Sqrt(1f - 0.4f * 0.4f),
				Mathf.Sqrt(1f - 0.6f * 0.6f),
				Mathf.Sqrt(1f - 0.8f * 0.8f),
				Mathf.Sqrt(1f - 0.2f * 0.2f - 0.2f * 0.2f),
				Mathf.Sqrt(1f - 0.2f * 0.2f - 0.4f * 0.4f),
				Mathf.Sqrt(1f - 0.2f * 0.2f - 0.6f * 0.6f),
				Mathf.Sqrt(1f - 0.2f * 0.2f - 0.8f * 0.8f),
				Mathf.Sqrt(1f - 0.4f * 0.4f - 0.4f * 0.4f),
				Mathf.Sqrt(1f - 0.4f * 0.4f - 0.6f * 0.6f),
				Mathf.Sqrt(1f - 0.4f * 0.4f - 0.8f * 0.8f),
				Mathf.Sqrt(1f - 0.6f * 0.6f - 0.6f * 0.6f)
			};

			private enum MipLevel { Original, L1, L2, L3, L4, L5, L6 }

			private enum Pass {
				DepthCopy,
				CompositionDeferred,
				CompositionForward,
				DebugOverlay
			}

			private static readonly int AOParams = Shader.PropertyToID("_AOParams");
			private static readonly int AOColor = Shader.PropertyToID("_AOColor");
			private static readonly int MSVOcclusionTexture = Shader.PropertyToID("_MSVOcclusionTexture");
			private static readonly int DepthCopy = Shader.PropertyToID("DepthCopy");
			private static readonly int LinearDepth = Shader.PropertyToID("LinearDepth");
			private static readonly int LowDepth1 = Shader.PropertyToID("LowDepth1");
			private static readonly int LowDepth2 = Shader.PropertyToID("LowDepth2");
			private static readonly int LowDepth3 = Shader.PropertyToID("LowDepth3");
			private static readonly int LowDepth4 = Shader.PropertyToID("LowDepth4");
			private static readonly int TiledDepth1 = Shader.PropertyToID("TiledDepth1");
			private static readonly int TiledDepth2 = Shader.PropertyToID("TiledDepth2");
			private static readonly int TiledDepth3 = Shader.PropertyToID("TiledDepth3");
			private static readonly int TiledDepth4 = Shader.PropertyToID("TiledDepth4");
			private static readonly int Occlusion1 = Shader.PropertyToID("Occlusion1");
			private static readonly int Occlusion2 = Shader.PropertyToID("Occlusion2");
			private static readonly int Occlusion3 = Shader.PropertyToID("Occlusion3");
			private static readonly int Occlusion4 = Shader.PropertyToID("Occlusion4");
			private static readonly int Combined1 = Shader.PropertyToID("Combined1");
			private static readonly int Combined2 = Shader.PropertyToID("Combined2");
			private static readonly int Combined3 = Shader.PropertyToID("Combined3");
			private static readonly int RenderViewportScaleFactor = Shader.PropertyToID("_RenderViewportScaleFactor");
			private static readonly int FogParams = Shader.PropertyToID("_FogParams");

			public AOPass(AOSettings settings) {
				this.settings = settings;
				this.renderPassEvent = settings.Event;

				this.blitMaterial = CoreUtils.CreateEngineMaterial(settings.shader);
				this.triangle = new Mesh();
				this.triangle.name = "Fullscreen Triangle";
				this.triangle.SetVertices(new[] {
					new Vector3(-1f, -1f, 0f),
					new Vector3(-1f,  3f, 0f),
					new Vector3( 3f, -1f, 0f)
				});
				this.triangle.SetIndices(new[] { 0, 1, 2 }, MeshTopology.Triangles, 0, false);
				this.triangle.UploadMeshData(true);
			}

			void Alloc(CommandBuffer cmd, int id, MipLevel size, RenderTextureFormat format, bool uav) {
				int sizeId = (int)size;
				cmd.GetTemporaryRT(id, new RenderTextureDescriptor {
					width = m_ScaledWidths[sizeId],
					height = m_ScaledHeights[sizeId],
					colorFormat = format,
					depthBufferBits = 0,
					volumeDepth = 1,
					autoGenerateMips = false,
					msaaSamples = 1,
					enableRandomWrite = uav,
					dimension = TextureDimension.Tex2D,
					sRGB = false
				}, FilterMode.Point);
			}

			void AllocArray(CommandBuffer cmd, int id, MipLevel size, RenderTextureFormat format, bool uav) {
				int sizeId = (int)size;
				cmd.GetTemporaryRT(id, new RenderTextureDescriptor {
					width = m_ScaledWidths[sizeId],
					height = m_ScaledHeights[sizeId],
					colorFormat = format,
					depthBufferBits = 0,
					volumeDepth = 16,
					autoGenerateMips = false,
					msaaSamples = 1,
					enableRandomWrite = uav,
					dimension = TextureDimension.Tex2DArray,
					sRGB = false
				}, FilterMode.Point);
			}

			// Calculate values in _ZBuferParams (built-in shader variable)
			// We can't use _ZBufferParams in compute shaders, so this function is
			// used to give the values in it to compute shaders.
			Vector4 CalculateZBufferParams(Camera camera) {
				float fpn = camera.farClipPlane / camera.nearClipPlane;

				if (SystemInfo.usesReversedZBuffer)
					return new Vector4(fpn - 1f, 1f, 0f, 0f);

				return new Vector4(1f - fpn, fpn, 0f, 0f);
			}

			float CalculateTanHalfFovHeight(Camera camera) {
				return 1f / camera.projectionMatrix[0, 0];
			}

			Vector2 GetSize(MipLevel mip) {
				return new Vector2(m_ScaledWidths[(int)mip], m_ScaledHeights[(int)mip]);
			}

			Vector3 GetSizeArray(MipLevel mip) {
				return new Vector3(m_ScaledWidths[(int)mip], m_ScaledHeights[(int)mip], 16);
			}

			private void PushAllocCommands(CommandBuffer cmd, bool isMSAA) {
				if (isMSAA) {
					Alloc(cmd, LinearDepth, MipLevel.Original, RenderTextureFormat.RGHalf, true);

					Alloc(cmd, LowDepth1, MipLevel.L1, RenderTextureFormat.RGFloat, true);
					Alloc(cmd, LowDepth2, MipLevel.L2, RenderTextureFormat.RGFloat, true);
					Alloc(cmd, LowDepth3, MipLevel.L3, RenderTextureFormat.RGFloat, true);
					Alloc(cmd, LowDepth4, MipLevel.L4, RenderTextureFormat.RGFloat, true);

					AllocArray(cmd, TiledDepth1, MipLevel.L3, RenderTextureFormat.RGHalf, true);
					AllocArray(cmd, TiledDepth2, MipLevel.L4, RenderTextureFormat.RGHalf, true);
					AllocArray(cmd, TiledDepth3, MipLevel.L5, RenderTextureFormat.RGHalf, true);
					AllocArray(cmd, TiledDepth4, MipLevel.L6, RenderTextureFormat.RGHalf, true);

					Alloc(cmd, Occlusion1, MipLevel.L1, RenderTextureFormat.RG16, true);
					Alloc(cmd, Occlusion2, MipLevel.L2, RenderTextureFormat.RG16, true);
					Alloc(cmd, Occlusion3, MipLevel.L3, RenderTextureFormat.RG16, true);
					Alloc(cmd, Occlusion4, MipLevel.L4, RenderTextureFormat.RG16, true);

					Alloc(cmd, Combined1, MipLevel.L1, RenderTextureFormat.RG16, true);
					Alloc(cmd, Combined2, MipLevel.L2, RenderTextureFormat.RG16, true);
					Alloc(cmd, Combined3, MipLevel.L3, RenderTextureFormat.RG16, true);
				} else {
					Alloc(cmd, LinearDepth, MipLevel.Original, RenderTextureFormat.RHalf, true);

					Alloc(cmd, LowDepth1, MipLevel.L1, RenderTextureFormat.RFloat, true);
					Alloc(cmd, LowDepth2, MipLevel.L2, RenderTextureFormat.RFloat, true);
					Alloc(cmd, LowDepth3, MipLevel.L3, RenderTextureFormat.RFloat, true);
					Alloc(cmd, LowDepth4, MipLevel.L4, RenderTextureFormat.RFloat, true);

					AllocArray(cmd, TiledDepth1, MipLevel.L3, RenderTextureFormat.RHalf, true);
					AllocArray(cmd, TiledDepth2, MipLevel.L4, RenderTextureFormat.RHalf, true);
					AllocArray(cmd, TiledDepth3, MipLevel.L5, RenderTextureFormat.RHalf, true);
					AllocArray(cmd, TiledDepth4, MipLevel.L6, RenderTextureFormat.RHalf, true);

					Alloc(cmd, Occlusion1, MipLevel.L1, RenderTextureFormat.R8, true);
					Alloc(cmd, Occlusion2, MipLevel.L2, RenderTextureFormat.R8, true);
					Alloc(cmd, Occlusion3, MipLevel.L3, RenderTextureFormat.R8, true);
					Alloc(cmd, Occlusion4, MipLevel.L4, RenderTextureFormat.R8, true);

					Alloc(cmd, Combined1, MipLevel.L1, RenderTextureFormat.R8, true);
					Alloc(cmd, Combined2, MipLevel.L2, RenderTextureFormat.R8, true);
					Alloc(cmd, Combined3, MipLevel.L3, RenderTextureFormat.R8, true);
				}
			}

			private void PushDownsampleCommands(CommandBuffer cmd, Camera camera, RenderTargetIdentifier? depthMap, bool isMSAA) {
				RenderTargetIdentifier depthMapId;

				if (depthMap != null) {
					depthMapId = depthMap.Value;
				} else {
					// TIPS: no supported DeferredRendering
					Alloc(cmd, DepthCopy, MipLevel.Original, RenderTextureFormat.RFloat, false);
					depthMapId = new RenderTargetIdentifier(DepthCopy);
					this.BlitFullscreenTriangle(cmd, BuiltinRenderTextureType.None, depthMapId, (int)Pass.DepthCopy, RenderBufferLoadAction.DontCare);
				}

				// 1st downsampling pass.
				var cs = this.settings.multiScaleAODownsample1;
				int kernel = cs.FindKernel(isMSAA ? "MultiScaleVODownsample1_MSAA" : "MultiScaleVODownsample1");

				cmd.SetComputeTextureParam(cs, kernel, "LinearZ", LinearDepth);
				cmd.SetComputeTextureParam(cs, kernel, "DS2x", LowDepth1);
				cmd.SetComputeTextureParam(cs, kernel, "DS4x", LowDepth2);
				cmd.SetComputeTextureParam(cs, kernel, "DS2xAtlas", TiledDepth1);
				cmd.SetComputeTextureParam(cs, kernel, "DS4xAtlas", TiledDepth2);
				cmd.SetComputeVectorParam(cs, "ZBufferParams", CalculateZBufferParams(camera));
				cmd.SetComputeTextureParam(cs, kernel, "Depth", depthMapId);

				cmd.DispatchCompute(cs, kernel, m_ScaledWidths[(int)MipLevel.L4], m_ScaledHeights[(int)MipLevel.L4], 1);
				cmd.ReleaseTemporaryRT(DepthCopy);

				// 2nd downsampling pass.
				cs = this.settings.multiScaleAODownsample2;
				kernel = isMSAA ? cs.FindKernel("MultiScaleVODownsample2_MSAA") : cs.FindKernel("MultiScaleVODownsample2");

				cmd.SetComputeTextureParam(cs, kernel, "DS4x", LowDepth2);
				cmd.SetComputeTextureParam(cs, kernel, "DS8x", LowDepth3);
				cmd.SetComputeTextureParam(cs, kernel, "DS16x", LowDepth4);
				cmd.SetComputeTextureParam(cs, kernel, "DS8xAtlas", TiledDepth3);
				cmd.SetComputeTextureParam(cs, kernel, "DS16xAtlas", TiledDepth4);

				cmd.DispatchCompute(cs, kernel, m_ScaledWidths[(int)MipLevel.L6], m_ScaledHeights[(int)MipLevel.L6], 1);
			}

			void PushRenderCommands(CommandBuffer cmd, int source, int destination, Vector3 sourceSize, float tanHalfFovH, bool isMSAA) {
				// Here we compute multipliers that convert the center depth value into (the reciprocal
				// of) sphere thicknesses at each sample location. This assumes a maximum sample radius
				// of 5 units, but since a sphere has no thickness at its extent, we don't need to
				// sample that far out. Only samples whole integer offsets with distance less than 25
				// are used. This means that there is no sample at (3, 4) because its distance is
				// exactly 25 (and has a thickness of 0.)

				// The shaders are set up to sample a circular region within a 5-pixel radius.
				const float kScreenspaceDiameter = 10f;

				// SphereDiameter = CenterDepth * ThicknessMultiplier. This will compute the thickness
				// of a sphere centered at a specific depth. The ellipsoid scale can stretch a sphere
				// into an ellipsoid, which changes the characteristics of the AO.
				// TanHalfFovH: Radius of sphere in depth units if its center lies at Z = 1
				// ScreenspaceDiameter: Diameter of sample sphere in pixel units
				// ScreenspaceDiameter / BufferWidth: Ratio of the screen width that the sphere actually covers
				float thicknessMultiplier = 2f * tanHalfFovH * kScreenspaceDiameter / sourceSize.x;
				//if (RuntimeUtilities.isSinglePassStereoEnabled)
				//    thicknessMultiplier *= 2f;

				// This will transform a depth value from [0, thickness] to [0, 1].
				float inverseRangeFactor = 1f / thicknessMultiplier;

				// The thicknesses are smaller for all off-center samples of the sphere. Compute
				// thicknesses relative to the center sample.
				for (int i = 0; i < 12; i++)
					m_InvThicknessTable[i] = inverseRangeFactor / m_SampleThickness[i];

				// These are the weights that are multiplied against the samples because not all samples
				// are equally important. The farther the sample is from the center location, the less
				// they matter. We use the thickness of the sphere to determine the weight.  The scalars
				// in front are the number of samples with this weight because we sum the samples
				// together before multiplying by the weight, so as an aggregate all of those samples
				// matter more. After generating this table, the weights are normalized.
				m_SampleWeightTable[0] = 4 * m_SampleThickness[0];    // Axial
				m_SampleWeightTable[1] = 4 * m_SampleThickness[1];    // Axial
				m_SampleWeightTable[2] = 4 * m_SampleThickness[2];    // Axial
				m_SampleWeightTable[3] = 4 * m_SampleThickness[3];    // Axial
				m_SampleWeightTable[4] = 4 * m_SampleThickness[4];    // Diagonal
				m_SampleWeightTable[5] = 8 * m_SampleThickness[5];    // L-shaped
				m_SampleWeightTable[6] = 8 * m_SampleThickness[6];    // L-shaped
				m_SampleWeightTable[7] = 8 * m_SampleThickness[7];    // L-shaped
				m_SampleWeightTable[8] = 4 * m_SampleThickness[8];    // Diagonal
				m_SampleWeightTable[9] = 8 * m_SampleThickness[9];    // L-shaped
				m_SampleWeightTable[10] = 8 * m_SampleThickness[10];    // L-shaped
				m_SampleWeightTable[11] = 4 * m_SampleThickness[11];    // Diagonal

				// Zero out the unused samples.
				// FIXME: should we support SAMPLE_EXHAUSTIVELY mode?
				m_SampleWeightTable[0] = 0;
				m_SampleWeightTable[2] = 0;
				m_SampleWeightTable[5] = 0;
				m_SampleWeightTable[7] = 0;
				m_SampleWeightTable[9] = 0;

				// Normalize the weights by dividing by the sum of all weights
				var totalWeight = 0f;

				foreach (float w in m_SampleWeightTable)
					totalWeight += w;

				for (int i = 0; i < m_SampleWeightTable.Length; i++)
					m_SampleWeightTable[i] /= totalWeight;

				// Set the arguments for the render kernel.
				var cs = this.settings.multiScaleAORender;
				int kernel = isMSAA ? cs.FindKernel("MultiScaleVORender_MSAA_interleaved") : cs.FindKernel("MultiScaleVORender_interleaved");

				cmd.SetComputeFloatParams(cs, "gInvThicknessTable", m_InvThicknessTable);
				cmd.SetComputeFloatParams(cs, "gSampleWeightTable", m_SampleWeightTable);
				cmd.SetComputeVectorParam(cs, "gInvSliceDimension", new Vector2(1f / sourceSize.x, 1f / sourceSize.y));
				cmd.SetComputeVectorParam(cs, "AdditionalParams", new Vector2(-1f / this.volumeComponent.thicknessModifier.value, this.volumeComponent.intensity.value));
				cmd.SetComputeTextureParam(cs, kernel, "DepthTex", source);
				cmd.SetComputeTextureParam(cs, kernel, "Occlusion", destination);

				// Calculate the thread group count and add a dispatch command with them.
				uint xsize, ysize, zsize;
				cs.GetKernelThreadGroupSizes(kernel, out xsize, out ysize, out zsize);

				cmd.DispatchCompute(
					cs, kernel,
					((int)sourceSize.x + (int)xsize - 1) / (int)xsize,
					((int)sourceSize.y + (int)ysize - 1) / (int)ysize,
					((int)sourceSize.z + (int)zsize - 1) / (int)zsize
				);
			}

			void PushUpsampleCommands(CommandBuffer cmd, int lowResDepth, int interleavedAO, int highResDepth, int? highResAO, RenderTargetIdentifier dest, Vector3 lowResDepthSize, Vector2 highResDepthSize, bool isMSAA, bool invert = false) {
				var cs = this.settings.multiScaleAOUpsample;
				int kernel = 0;
				if (!isMSAA) {
					kernel = cs.FindKernel(highResAO == null ? invert
						? "MultiScaleVOUpSample_invert"
						: "MultiScaleVOUpSample"
						: "MultiScaleVOUpSample_blendout");
				} else {
					kernel = cs.FindKernel(highResAO == null ? invert
					? "MultiScaleVOUpSample_MSAA_invert"
					: "MultiScaleVOUpSample_MSAA"
					: "MultiScaleVOUpSample_MSAA_blendout");
				}


				float stepSize = 1920f / lowResDepthSize.x;
				float bTolerance = 1f - Mathf.Pow(10f, this.volumeComponent.blurTolerance.value) * stepSize;
				bTolerance *= bTolerance;
				float uTolerance = Mathf.Pow(10f, this.volumeComponent.upsampleTolerance.value);
				float noiseFilterWeight = 1f / (Mathf.Pow(10f, this.volumeComponent.noiseFilterTolerance.value) + uTolerance);

				cmd.SetComputeVectorParam(cs, "InvLowResolution", new Vector2(1f / lowResDepthSize.x, 1f / lowResDepthSize.y));
				cmd.SetComputeVectorParam(cs, "InvHighResolution", new Vector2(1f / highResDepthSize.x, 1f / highResDepthSize.y));
				cmd.SetComputeVectorParam(cs, "AdditionalParams", new Vector4(noiseFilterWeight, stepSize, bTolerance, uTolerance));

				cmd.SetComputeTextureParam(cs, kernel, "LoResDB", lowResDepth);
				cmd.SetComputeTextureParam(cs, kernel, "HiResDB", highResDepth);
				cmd.SetComputeTextureParam(cs, kernel, "LoResAO1", interleavedAO);

				if (highResAO != null)
					cmd.SetComputeTextureParam(cs, kernel, "HiResAO", highResAO.Value);

				cmd.SetComputeTextureParam(cs, kernel, "AoResult", dest);

				int xcount = ((int)highResDepthSize.x + 17) / 16;
				int ycount = ((int)highResDepthSize.y + 17) / 16;
				cmd.DispatchCompute(cs, kernel, xcount, ycount, 1);
			}

			void PushReleaseCommands(CommandBuffer cmd) {
				cmd.ReleaseTemporaryRT(LinearDepth);

				cmd.ReleaseTemporaryRT(LowDepth1);
				cmd.ReleaseTemporaryRT(LowDepth2);
				cmd.ReleaseTemporaryRT(LowDepth3);
				cmd.ReleaseTemporaryRT(LowDepth4);

				cmd.ReleaseTemporaryRT(TiledDepth1);
				cmd.ReleaseTemporaryRT(TiledDepth2);
				cmd.ReleaseTemporaryRT(TiledDepth3);
				cmd.ReleaseTemporaryRT(TiledDepth4);

				cmd.ReleaseTemporaryRT(Occlusion1);
				cmd.ReleaseTemporaryRT(Occlusion2);
				cmd.ReleaseTemporaryRT(Occlusion3);
				cmd.ReleaseTemporaryRT(Occlusion4);

				cmd.ReleaseTemporaryRT(Combined1);
				cmd.ReleaseTemporaryRT(Combined2);
				cmd.ReleaseTemporaryRT(Combined3);
			}

			private void GenerateAOMap(CommandBuffer cmd, Camera camera, int destination, RenderTargetIdentifier? depthMap, bool invert, bool isMSAA) {
				// Base size
				//m_ScaledWidths[0] = camera.scaledPixelWidth * (RuntimeUtilities.isSinglePassStereoEnabled ? 2 : 1); // TIPS: No supported XR
				m_ScaledWidths[0] = camera.scaledPixelWidth;
				m_ScaledHeights[0] = camera.scaledPixelHeight;

				// L1 -> L6 sizes
				for (int i = 1; i < 7; i++) {
					int div = 1 << i;
					m_ScaledWidths[i] = (m_ScaledWidths[0] + (div - 1)) / div;
					m_ScaledHeights[i] = (m_ScaledHeights[0] + (div - 1)) / div;
				}

				// Allocate temporary textures
				PushAllocCommands(cmd, isMSAA);

				// Render logic
				PushDownsampleCommands(cmd, camera, depthMap, isMSAA);

				float tanHalfFovH = CalculateTanHalfFovHeight(camera);
				PushRenderCommands(cmd, TiledDepth1, Occlusion1, GetSizeArray(MipLevel.L3), tanHalfFovH, isMSAA);
				PushRenderCommands(cmd, TiledDepth2, Occlusion2, GetSizeArray(MipLevel.L4), tanHalfFovH, isMSAA);
				PushRenderCommands(cmd, TiledDepth3, Occlusion3, GetSizeArray(MipLevel.L5), tanHalfFovH, isMSAA);
				PushRenderCommands(cmd, TiledDepth4, Occlusion4, GetSizeArray(MipLevel.L6), tanHalfFovH, isMSAA);

				PushUpsampleCommands(cmd, LowDepth4, Occlusion4, LowDepth3, Occlusion3, Combined3, GetSize(MipLevel.L4), GetSize(MipLevel.L3), isMSAA);
				PushUpsampleCommands(cmd, LowDepth3, Combined3, LowDepth2, Occlusion2, Combined2, GetSize(MipLevel.L3), GetSize(MipLevel.L2), isMSAA);
				PushUpsampleCommands(cmd, LowDepth2, Combined2, LowDepth1, Occlusion1, Combined1, GetSize(MipLevel.L2), GetSize(MipLevel.L1), isMSAA);
				PushUpsampleCommands(cmd, LowDepth1, Combined1, LinearDepth, null, destination, GetSize(MipLevel.L1), GetSize(MipLevel.Original), isMSAA, invert);

				// Cleanup
				PushReleaseCommands(cmd);
			}

			public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData) {
#if UNITY_EDITOR || DEBUG
				if (renderingData.cameraData.isSceneViewCamera)
					return;
				if (renderingData.cameraData.camera.cameraType == CameraType.Preview)
					return;
				if (this.blitMaterial == null)
					return;
#endif

				this.volumeComponent = VolumeManager.instance.stack.GetComponent<fAmbientOcclusion>();
				if (!this.volumeComponent.IsActive())
					return;

				CommandBuffer cmd = CommandBufferPool.Get(PASS_NAME);

				using (new ProfilingScope(cmd, profilingSampler)) {
					cmd.Clear();

					Shader.SetGlobalFloat(RenderViewportScaleFactor, 1.0f);

					var camera = renderingData.cameraData.camera;
					var rtDescriptor = renderingData.cameraData.cameraTargetDescriptor;

					cmd.GetTemporaryRT(AO_TARGET,
						new RenderTextureDescriptor {
							width = rtDescriptor.width,
							height = rtDescriptor.height,
							colorFormat = RenderTextureFormat.R8,
							depthBufferBits = 0,
							volumeDepth = 1,
							autoGenerateMips = false,
							msaaSamples = 1,
							enableRandomWrite = true,
							dimension = TextureDimension.Tex2D,
							sRGB = false
						}, FilterMode.Point);

					// WIP: ScalableAO------------------------------------------------------
					//// Material setup
					//// Always use a quater-res AO buffer unless High/Ultra quality is set.
					//bool downsampling = (int)this.volumeComponent.quality.value < (int)AmbientOcclusionQuality.High;
					//float px = this.volumeComponent.intensity.value;
					//float py = this.volumeComponent.radius.value;
					//float pz = downsampling ? 0.5f : 1f;
					//float pw = m_SampleCount[(int)this.volumeComponent.quality.value];
					//this.blitMaterial.SetVector(AOParams, new Vector4(px, py, pz, pw));
					//-----------------------------------------------------------------------
					this.blitMaterial.SetVector(AOColor, Color.white - this.volumeComponent.color.value);

					// In Forward mode, fog is applied at the object level in the grometry pass so we need
					// to apply it to AO as well or it'll drawn on top of the fog effect.
					if (camera.actualRenderingPath == RenderingPath.Forward && RenderSettings.fog) {
						this.blitMaterial.EnableKeyword("APPLY_FORWARD_FOG");
						this.blitMaterial.SetVector(FogParams, new Vector3(RenderSettings.fogDensity, RenderSettings.fogStartDistance, RenderSettings.fogEndDistance)
						);
					}

					GenerateAOMap(cmd, camera, AO_TARGET, null, false, false);
					cmd.SetGlobalTexture(MSVOcclusionTexture, AO_TARGET);
					this.BlitFullscreenTriangle(cmd, BuiltinRenderTextureType.None, this.colorAttachment, (int)Pass.CompositionForward, RenderBufferLoadAction.Load);
					cmd.ReleaseTemporaryRT(AO_TARGET);
					context.ExecuteCommandBuffer(cmd);
				}

				CommandBufferPool.Release(cmd);
			}

			private void BlitFullscreenTriangle(CommandBuffer cmd, RenderTargetIdentifier source, RenderTargetIdentifier destination, int pass, RenderBufferLoadAction loadAction) {
				cmd.SetGlobalTexture(MAIN_TEX, source);
				cmd.SetRenderTarget(destination, loadAction, RenderBufferStoreAction.Store);
				cmd.DrawMesh(this.triangle, Matrix4x4.identity, this.blitMaterial, 0, pass);
			}
		}
		#endregion

		public override void Create() {
#if UNITY_EDITOR
			if (this.settings.shader == null || this.settings.multiScaleAODownsample1 == null) {
				this.settings.shader = Shader.Find("Hidden/PostProcessing/MultiScaleVO");

				var ppRes = UnityEditor.AssetDatabase.LoadAssetAtPath<UnityEngine.Rendering.PostProcessing.PostProcessResources>("Packages/com.unity.postprocessing/PostProcessing/PostProcessResources.asset");
				if (ppRes == null)
					return;
				this.settings.multiScaleAODownsample1 = ppRes.computeShaders.multiScaleAODownsample1;
				this.settings.multiScaleAODownsample2 = ppRes.computeShaders.multiScaleAODownsample2;
				this.settings.multiScaleAORender = ppRes.computeShaders.multiScaleAORender;
				this.settings.multiScaleAOUpsample = ppRes.computeShaders.multiScaleAOUpsample;

				this.settings.LayerMask = LayerMask.NameToLayer("Everything");
			}

			UniversalRenderPipeline.asset.supportsCameraDepthTexture = true;    // NOTE: Require DepthTexture
#endif

			this.aoPass = new AOPass(this.settings);
		}

		public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData) {
			this.aoPass.ConfigureTarget(renderer.cameraColorTarget, renderer.cameraDepth);
			renderer.EnqueuePass(this.aoPass);
		}
	}
}
