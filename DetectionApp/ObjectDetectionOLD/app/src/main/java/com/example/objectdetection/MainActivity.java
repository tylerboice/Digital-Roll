package com.example.objectdetection;


import androidx.appcompat.app.AppCompatActivity;


import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;

import android.content.pm.PackageManager;
import android.hardware.Camera;

import android.os.Build;
import android.os.Bundle;

import android.view.OrientationEventListener;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import android.widget.FrameLayout;


import java.io.IOException;


public class MainActivity extends AppCompatActivity {

    private boolean isCameraInitialized;

    private Camera mCamera = null;

    private static SurfaceHolder myHolder;

    private static CameraPreview mPreview;

    private FrameLayout preview;

    private static OrientationEventListener orientationEventListener = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    private static final int REQUEST_PERMISSIONS = 34;

    private static final int PERMISSIONS_COUNT = 1;

    private static final String[] PERMISSIONS = {
            Manifest.permission.CAMERA
    };

    @SuppressLint("NewApi")
    private boolean arePermissionsDenied(){
        for(int i = 0; i< PERMISSIONS_COUNT ; i++){
            if(checkSelfPermission(PERMISSIONS[i])!= PackageManager.PERMISSION_GRANTED){
                return true;
            }
        }
        return false;
    }
    @Override
    protected void onResume(){
        super.onResume();
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && arePermissionsDenied()){
            requestPermissions(PERMISSIONS, REQUEST_PERMISSIONS);
            return;
        }

        if(!isCameraInitialized) {
            mCamera = Camera.open();
            mPreview = new CameraPreview(this, mCamera);
            preview = findViewById(R.id.camera_preview);
            preview.addView(mPreview);

            orientationEventListener = new OrientationEventListener(this) {
                @Override
                public void onOrientationChanged(int orientation) {
                    rotateCamera();
                }
            };
            orientationEventListener.enable();
        }




    }

    private static int rotation;

    private static boolean whichCamera = true;

    private static Camera.Parameters p;

    private void rotateCamera(){
        if(mCamera!=null) {
            rotation = this.getWindowManager().getDefaultDisplay().getRotation();
            if(rotation == 0){
                rotation = 90;
            }
            else if(rotation == 1){
                rotation = 0;
            }
            else if(rotation == 2){
                rotation = 270;
            }
            else{
                rotation = 180;
            }
            mCamera.setDisplayOrientation(rotation);
            if(!whichCamera){
                if(rotation == 90){
                    rotation = 270;
                }
                else if(rotation == 270) {
                    rotation = 90;
                }
            }
            p = mCamera.getParameters();
            p.setRotation(rotation);
            mCamera.setParameters(p);
        }
    }

    private static class CameraPreview extends SurfaceView implements SurfaceHolder.Callback{
        private static SurfaceHolder mHolder;
        private static Camera mCamera;

        private CameraPreview(Context context, Camera camera){
            super(context);
            mCamera = camera;
            mHolder = getHolder();
            mHolder.addCallback(this);
            mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        }

        public void surfaceCreated(SurfaceHolder holder){
            myHolder = holder;
            try {
                mCamera.setPreviewDisplay(holder);
                mCamera.startPreview();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public void surfaceDestroyed(SurfaceHolder holder){

        }

        public void surfaceChanged(SurfaceHolder holder, int format, int w, int h){

        }
    }

}
