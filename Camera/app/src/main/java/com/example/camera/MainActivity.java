package com.example.camera;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.OrientationEventListener;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.content.Context;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.List;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private SensorManager mSensorManager;
    private Sensor mRotationSensor;

    private static final int SENSOR_DELAY = 500 * 1000; // 500ms
    private static final int FROM_RADS_TO_DEGS = -57;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            mSensorManager = (SensorManager) getSystemService(Activity.SENSOR_SERVICE);
            mRotationSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
            mSensorManager.registerListener(this, mRotationSensor, SENSOR_DELAY);
        } catch (Exception e) {
            Toast.makeText(this, "Hardware compatibility issue", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor == mRotationSensor) {
            if (event.values.length > 4) {
                float[] truncatedRotationVector = new float[4];
                System.arraycopy(event.values, 0, truncatedRotationVector, 0, 4);
                update(truncatedRotationVector);
            } else {
                update(event.values);
            }
        }
    }

    private void update(float[] vectors) {
        float[] rotationMatrix = new float[9];
        SensorManager.getRotationMatrixFromVector(rotationMatrix, vectors);
        int worldAxisX = SensorManager.AXIS_X;
        int worldAxisZ = SensorManager.AXIS_Z;
        float[] adjustedRotationMatrix = new float[9];
        SensorManager.remapCoordinateSystem(rotationMatrix, worldAxisX, worldAxisZ, adjustedRotationMatrix);
        float[] orientation = new float[3];
        SensorManager.getOrientation(rotationMatrix, orientation);
        float pitch = orientation[0] * FROM_RADS_TO_DEGS;
        float yaw = orientation[1] * FROM_RADS_TO_DEGS;
        float roll = orientation[2] * FROM_RADS_TO_DEGS;
        ((TextView)findViewById(R.id.pitch)).setText("Pitch: "+yaw);
        ((TextView)findViewById(R.id.yaw)).setText("Yaw: "+pitch);
        ((TextView)findViewById(R.id.roll)).setText("Roll: "+roll);
    }

    private static final String[] PERMISSIONS = {
            Manifest.permission.CAMERA
    };

    private static final int REQUEST_PERMISSIONS = 34;

    private static final int PERMISSIONS_COUNT = 1;

    @SuppressLint("NewApi")
    private boolean arePermissionsDenied(){
        for(int i = 0; i< PERMISSIONS_COUNT ; i++){
            if(checkSelfPermission(PERMISSIONS[i])!= PackageManager.PERMISSION_GRANTED){
                return true;
            }
        }
        return false;
    }

    @SuppressLint("NewApi")
    //@Override
    public void onRequestPermissionResult(int requestCode, String[] permissions, int[] grantResults){
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if(requestCode == REQUEST_PERMISSIONS && grantResults.length > 0){
            if(arePermissionsDenied()){
                ((ActivityManager) (this.getSystemService(ACTIVITY_SERVICE))).clearApplicationUserData();
                recreate();
            }
            else{
                onResume();
            }
        }
    }

    private boolean isCameraInitialized;

    private Camera mCamera = null;

    private static SurfaceHolder myHolder;

    private static CameraPreview mPreview;

    private FrameLayout preview;

    private Button flashB;

    private static OrientationEventListener orientationEventListener = null;

    private static boolean fM;


    @Override
    protected void onResume(){
        super.onResume();
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && arePermissionsDenied()){
            requestPermissions(PERMISSIONS, REQUEST_PERMISSIONS);
            return;
        }
        if(!isCameraInitialized){
            mCamera = Camera.open();
            mPreview = new CameraPreview(this, mCamera);
            preview = findViewById(R.id.camera_preview);
            preview.addView(mPreview);
            rotateCamera();
            flashB = findViewById(R.id.flash);
            if(hasFlash()){
                flashB.setVisibility(View.GONE);
            }
            else{
                flashB.setVisibility((View.GONE));
            }
            final File myimagesFolder = new File(getExternalFilesDir(null), "MyImages");
            myimagesFolder.mkdirs();
            final Button switchCameraButton = findViewById(R.id.switchCamera);
            switchCameraButton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {

                    Intent camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivity(camera);
                    mCamera.release();

                }
            });
            orientationEventListener = new OrientationEventListener(this) {
                @Override
                public void onOrientationChanged(int orientation) {
                    rotateCamera();
                }
            };
            orientationEventListener.enable();
            preview.setOnLongClickListener(new View.OnLongClickListener(){
                @Override
                public boolean onLongClick(View v){
                 if(whichCamera){
                     if(fM){
                         p.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);
                     }
                     else {
                         p.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
                     }
                     try {
                         mCamera.setParameters(p);
                     }
                     catch (Exception e){

                     }
                     fM = !fM;
                 }
                 return true;
                }
            });
        }
    }

    private void switchCamera(){
        if(whichCamera){
            mCamera = Camera.open(Camera.CameraInfo.CAMERA_FACING_FRONT);
        }
        else{
            mCamera = Camera.open();
        }
        whichCamera = !whichCamera;
    }

    @Override
    protected void onPause(){
        super.onPause();
        releaseCamera();
    }

    private void releaseCamera(){
        if(mCamera != null){
            preview.removeView(mPreview);
            mCamera.release();
            orientationEventListener.disable();
            mCamera = null;
            whichCamera = !whichCamera;
        }
    }

    private static List<String> camEffects;

    private static boolean hasFlash(){
        camEffects = p.getSupportedColorEffects();
        final List<String> flashModes = p.getSupportedFlashModes();
        if(flashModes == null){
            return false;
        }
        for(String flashMode:flashModes){
            if(Camera.Parameters.FLASH_MODE_ON.equals(flashMode)){
                return true;
            }
        }
        return false;
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
