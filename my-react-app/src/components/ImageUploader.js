import React, { useState } from 'react';
import axios from 'axios';

const ImageUploader = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [result, setResult] = useState('');
    const [imagePreview, setImagePreview] = useState(''); // 画像プレビュー用のステートを追加

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        
        // 画像のプレビューを設定
        const reader = new FileReader();
        reader.onloadend = () => {
            setImagePreview(reader.result);
        };
        reader.readAsDataURL(file);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await axios.post('http://localhost:5000/', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setResult(response.data.result);
        } catch (error) {
            console.error('Error uploading file:', error);
            setResult('予測に失敗しました。');
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input type="file" accept="image/*" onChange={handleFileChange} required />
                <button type="submit">予測</button>
            </form>
            {imagePreview && (
                <div>
                    <h4>アップロードした画像:</h4>
                    <img src={imagePreview} alt="Uploaded" style={{ width: '300px', height: 'auto' }} />
                </div>
            )}
            {result && <h2>予測結果は... {result} !?</h2>}
        </div>
    );
};

export default ImageUploader;
