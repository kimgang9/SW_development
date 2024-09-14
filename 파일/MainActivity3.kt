package com.example.myapplication

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.storage.FirebaseStorage
import java.text.SimpleDateFormat
import java.util.Date
import java.util.UUID

class MainActivity3 : AppCompatActivity() {

    private var fbStorage: FirebaseStorage? = null
    private var firestore: FirebaseFirestore? = null
    private val uriPhotoMap = mutableMapOf<Int, Uri>()
    private var currentImageViewId: Int? = null

    private lateinit var photoPickerLauncher: ActivityResultLauncher<Intent>
    private lateinit var permissionLauncher: ActivityResultLauncher<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main3)

        fbStorage = FirebaseStorage.getInstance()
        firestore = FirebaseFirestore.getInstance()

        permissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                pickPhotoFromGallery()
            } else {
                Toast.makeText(this, "사진을 업로드할 수 없습니다.", Toast.LENGTH_LONG).show()
            }
        }

        photoPickerLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                val uri = result.data?.data
                currentImageViewId?.let { imageViewId ->
                    uri?.let {
                        uriPhotoMap[imageViewId] = it
                        val imageView: ImageView = findViewById(imageViewId)
                        imageView.setImageURI(it)
                    }
                }
            }
        }

        setupButtonListeners()
    }

    private fun setupButtonListeners() {
        setupImageButton(R.id.select_front_btn, R.id.front_img, "post_use")
        setupImageButton(R.id.select_driver_front_btn, R.id.driver_front_img, "post_use")
        setupImageButton(R.id.select_driver_rear_btn, R.id.driver_rear_img, "post_use")
        setupImageButton(R.id.select_passenger_front_btn, R.id.passenger_front_img, "post_use")
        setupImageButton(R.id.select_passenger_rear_btn, R.id.passenger_rear_img, "post_use")
        setupImageButton(R.id.select_rear_btn, R.id.rear_img, "post_use")

//        findViewById<Button>(R.id.btn_page_2).setOnClickListener {
//            val intent = Intent(this, MainActivity4::class.java)
//
//            // Firestore에서 이미지 메타데이터를 가져와서 Intent에 추가
//            firestore?.collection("image_metadata")?.get()?.addOnSuccessListener { result ->
//                result.forEach { document ->
//                    val data = document.data
//                    val fileName = data["fileName"] as? String ?: return@forEach
//                    val imageType = data["imageType"] as? String ?: return@forEach
//
//                    // 메타데이터에 따라 이미지 URL 구성
//                    val imageUrl = "https://your-storage-bucket-url/${data["photoType"]}/$fileName"
//                    intent.putExtra("${imageType}_IMAGE_URL", imageUrl)
//                }
//
//                // MainActivity4로 Intent로 데이터 전달
//                startActivity(intent)
//            }?.addOnFailureListener {
//                Toast.makeText(this, "메타데이터 로드 실패", Toast.LENGTH_SHORT).show()
//            }
//        }
    }

    private fun setupImageButton(selectButtonId: Int, imageViewId: Int, photoType: String) {
        findViewById<View>(selectButtonId).setOnClickListener {
            currentImageViewId = imageViewId
            pickPhotoFromGallery()
        }

        val uploadButtonId = getUploadButtonId(selectButtonId)
        uploadButtonId?.let {
            findViewById<View>(it).setOnClickListener {
                uploadImage(imageViewId, photoType)
            }
        } ?: run {
            Toast.makeText(this, "알 수 없는 버튼 ID", Toast.LENGTH_SHORT).show()
        }
    }

    private fun getUploadButtonId(selectButtonId: Int): Int? {
        return when (selectButtonId) {
            R.id.select_front_btn -> R.id.upload_front_btn
            R.id.select_driver_front_btn -> R.id.upload_driver_front_btn
            R.id.select_driver_rear_btn -> R.id.upload_driver_rear_btn
            R.id.select_passenger_front_btn -> R.id.upload_passenger_front_btn
            R.id.select_passenger_rear_btn -> R.id.upload_passenger_rear_btn
            R.id.select_rear_btn -> R.id.upload_rear_btn
            else -> null
        }
    }

    private fun pickPhotoFromGallery() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            permissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
        } else {
            val photoPickerIntent = Intent(Intent.ACTION_PICK).apply {
                type = "image/*"
            }
            photoPickerLauncher.launch(photoPickerIntent)
        }
    }

    private fun uploadImage(imageViewId: Int, photoType: String) {
        val uri = uriPhotoMap[imageViewId]
        uri?.let {
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
            val imageFileName = "IMAGE_${UUID.randomUUID()}.png"
            val storageRef = fbStorage?.reference?.child("$photoType/$imageFileName")

            storageRef?.putFile(it)?.addOnProgressListener { taskSnapshot ->
                val progress = (100.0 * taskSnapshot.bytesTransferred / taskSnapshot.totalByteCount).toInt()
                Toast.makeText(this, "업로드 진행 중: $progress%", Toast.LENGTH_SHORT).show()
            }?.addOnSuccessListener {
                Toast.makeText(this, "이미지 업로드 완료", Toast.LENGTH_SHORT).show()
                saveImageMetadata(imageFileName, imageViewId, photoType)
            }?.addOnFailureListener {
                Toast.makeText(this, "업로드 실패", Toast.LENGTH_SHORT).show()
            }
        } ?: run {
            Toast.makeText(this, "이미지 URI가 비어 있습니다.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun saveImageMetadata(imageFileName: String, imageViewId: Int, photoType: String) {
        val imageType = when (imageViewId) {
            R.id.front_img -> "front"
            R.id.driver_front_img -> "driver_front"
            R.id.driver_rear_img -> "driver_rear"
            R.id.passenger_front_img -> "passenger_front"
            R.id.passenger_rear_img -> "passenger_rear"
            R.id.rear_img -> "rear"
            else -> "unknown"
        }

        val imageMetadata = mapOf(
            "fileName" to imageFileName,
            "imageType" to imageType,
            "photoType" to photoType,
            "timestamp" to System.currentTimeMillis()
        )

        firestore?.collection("image_metadata")?.add(imageMetadata)
            ?.addOnSuccessListener {
                Toast.makeText(this, "데이터 저장 성공", Toast.LENGTH_SHORT).show()
            }
            ?.addOnFailureListener {
                Toast.makeText(this, "데이터 저장 실패", Toast.LENGTH_SHORT).show()
            }
    }
}
