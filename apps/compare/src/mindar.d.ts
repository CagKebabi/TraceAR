/** Minimal typings for the MindAR ESM dist files (no official types). */

declare module "mind-ar/dist/mindar-image-three.prod.js" {
  export interface MindARAnchor {
    group: import("three").Group;
    onTargetFound?: () => void;
    onTargetLost?: () => void;
  }
  export class MindARThree {
    constructor(options: {
      container: HTMLElement;
      imageTargetSrc: string;
      uiScanning?: string;
      uiLoading?: string;
      uiError?: string;
      maxTrack?: number;
      filterMinCF?: number;
      filterBeta?: number;
    });
    renderer: import("three").WebGLRenderer;
    scene: import("three").Scene;
    camera: import("three").PerspectiveCamera;
    video: HTMLVideoElement;
    addAnchor(index: number): MindARAnchor;
    start(): Promise<void>;
    stop(): void;
  }
}

declare module "mind-ar/dist/mindar-image.prod.js" {
  export class Compiler {
    compileImageTargets(images: HTMLImageElement[], onProgress?: (p: number) => void): Promise<unknown>;
    exportData(): Promise<ArrayBuffer>;
  }
}
