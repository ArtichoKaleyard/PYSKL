# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import numpy as np

from pyskl.utils import warning_r0
from ..builder import PIPELINES


@PIPELINES.register_module()
class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        seed (int): The random seed used during test time. Default: 255.
    """

    def __init__(self,
                 clip_len,
                 num_clips=1,
                 p_interval=1,
                 seed=255,
                 **deprecated_kwargs):

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)
        if len(deprecated_kwargs):
            warning_r0('[UniformSampleFrames] The following args has been deprecated: ')
            for k, v in deprecated_kwargs.items():
                warning_r0(f'Arg name: {k}; Arg value: {v}')

    def _get_train_clips(self, num_frames, clip_len):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        for clip_idx in range(self.num_clips):
            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            inds = inds + off
            num_frames = old_num_frames

            allinds.append(inds)

        return np.concatenate(allinds)

    def _get_test_clips(self, num_frames, clip_len):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        np.random.seed(self.seed)

        all_inds = []

        for i in range(self.num_clips):

            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds + off)
            num_frames = old_num_frames

        return np.concatenate(all_inds)

    def __call__(self, results):
        num_frames = results['total_frames']

        if results.get('test_mode', False):
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results['start_index']
        inds = inds + start_index

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(int)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results['frame_inds'] = inds.astype(int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'seed={self.seed})')
        return repr_str


@PIPELINES.register_module()
class UniformSample(UniformSampleFrames):
    pass


@PIPELINES.register_module()
class KnsSampleFrames(UniformSampleFrames):
    """Key-neighborhood sampler for test-time pose clips.

    The sampler keeps the PoseC3D backbone and heatmap generation unchanged.
    It only replaces the temporal indices selected before ``PoseDecode``.  Each
    clip is split into coarse partitions; every partition keeps the strongest
    velocity and acceleration events, merges close events, and fills the
    remaining slot from the longest uncovered interval.

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of KNS clips to sample.
        p_interval (float | tuple[float, float]): Temporal crop ratio range.
            The first KNS version is intended for full-video test-time use, so
            the default is ``1``.
        seed (int): Test-time random seed used by optional uniform companion
            clips.
        frames_per_partition (int): Number of sampled frames per partition.
            KNS-v1 uses velocity, acceleration and one fill point, so this
            defaults to ``3``.
        merge_threshold (int): Maximum temporal distance for merging velocity
            and acceleration peaks into one event.
        uniform_clips (int): Optional number of leading uniform clips.  This is
            used for the ``uniform + KNS`` low-cost ensemble.
        smooth_kernel (int): Temporal moving-average width applied after
            confidence weighting.
        eps (float): Small value for robust normalization.
    """

    def __init__(
            self,
            clip_len,
            num_clips=1,
            p_interval=1,
            seed=255,
            frames_per_partition=3,
            merge_threshold=2,
            uniform_clips=0,
            smooth_kernel=3,
            eps=1e-6,
            **kwargs):
        super().__init__(clip_len, num_clips=num_clips, p_interval=p_interval, seed=seed, **kwargs)
        if clip_len % frames_per_partition != 0:
            raise ValueError('clip_len must be divisible by frames_per_partition.')
        if frames_per_partition != 3:
            raise ValueError('KNS-v1 expects exactly three frames per partition.')
        self.frames_per_partition = frames_per_partition
        self.merge_threshold = merge_threshold
        self.uniform_clips = uniform_clips
        self.smooth_kernel = smooth_kernel
        self.eps = eps

    @staticmethod
    def _midpoint(start, end):
        """Return the integer midpoint of a half-open interval."""

        if end <= start:
            return start
        return (start + end - 1) // 2

    def _robust_normalize(self, values):
        """Normalize a temporal signal with P5/P95 clipping."""

        values = values.astype(np.float32)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros_like(values, dtype=np.float32)
        lo, hi = np.percentile(finite, [5, 95])
        scale = hi - lo
        if scale <= self.eps:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip((values - lo) / (scale + self.eps), 0, 1).astype(np.float32)

    def _smooth(self, values):
        """Apply a short moving average without changing signal length."""

        if self.smooth_kernel <= 1 or values.size <= 1:
            return values
        kernel = np.ones(self.smooth_kernel, dtype=np.float32) / self.smooth_kernel
        pad_left = self.smooth_kernel // 2
        pad_right = self.smooth_kernel - 1 - pad_left
        padded = np.pad(values, (pad_left, pad_right), mode='edge')
        return np.convolve(padded, kernel, mode='valid').astype(np.float32)

    def _confidence(self, keypoint, keypoint_score=None):
        """Estimate per-frame pose reliability."""

        if keypoint_score is not None:
            score = keypoint_score.astype(np.float32)
            valid = score > self.eps
            denom = np.maximum(valid.sum(axis=(0, 2)), 1)
            return (score * valid).sum(axis=(0, 2)) / denom

        coords = keypoint[..., :2].astype(np.float32)
        valid = np.abs(coords).sum(axis=(-1, -2)) > self.eps
        denom = np.maximum(valid.sum(axis=0), 1)
        return valid.sum(axis=0).astype(np.float32) / denom

    def _motion_signals(self, keypoint, keypoint_score=None):
        """Compute confidence-weighted velocity and acceleration signals."""

        total_frames = keypoint.shape[1]
        coords = keypoint[..., :2].astype(np.float32)
        confidence = self._confidence(keypoint, keypoint_score)

        velocity = np.zeros(total_frames, dtype=np.float32)
        acceleration = np.zeros(total_frames, dtype=np.float32)
        if total_frames <= 1:
            return velocity, acceleration, confidence

        diffs = coords[:, 1:] - coords[:, :-1]
        speed = np.linalg.norm(diffs, axis=-1)
        if keypoint_score is not None:
            pair_score = np.minimum(keypoint_score[:, 1:], keypoint_score[:, :-1]).astype(np.float32)
            denom = np.maximum((pair_score > self.eps).sum(axis=(0, 2)), 1)
            velocity[1:] = (speed * pair_score).sum(axis=(0, 2)) / denom
        else:
            valid_pair = np.abs(coords[:, 1:]).sum(axis=(-1, -2)) > self.eps
            valid_pair &= np.abs(coords[:, :-1]).sum(axis=(-1, -2)) > self.eps
            denom = np.maximum(valid_pair.sum(axis=0), 1)
            velocity[1:] = (speed.sum(axis=2) * valid_pair).sum(axis=0) / denom

        if total_frames > 2:
            accel_vec = diffs[:, 1:] - diffs[:, :-1]
            accel = np.linalg.norm(accel_vec, axis=-1)
            if keypoint_score is not None:
                triple_score = np.minimum.reduce((
                    keypoint_score[:, 2:],
                    keypoint_score[:, 1:-1],
                    keypoint_score[:, :-2],
                )).astype(np.float32)
                denom = np.maximum((triple_score > self.eps).sum(axis=(0, 2)), 1)
                acceleration[2:] = (accel * triple_score).sum(axis=(0, 2)) / denom
            else:
                valid_triple = np.abs(coords[:, 2:]).sum(axis=(-1, -2)) > self.eps
                valid_triple &= np.abs(coords[:, 1:-1]).sum(axis=(-1, -2)) > self.eps
                valid_triple &= np.abs(coords[:, :-2]).sum(axis=(-1, -2)) > self.eps
                denom = np.maximum(valid_triple.sum(axis=0), 1)
                acceleration[2:] = (accel.sum(axis=2) * valid_triple).sum(axis=0) / denom

        velocity = self._smooth(self._robust_normalize(velocity) * confidence)
        acceleration = self._smooth(self._robust_normalize(acceleration) * confidence)
        return velocity, acceleration, confidence

    def _segment_peak(self, signal, start, end):
        """Find the strongest frame in a half-open segment."""

        if end <= start:
            return start
        segment = signal[start:end]
        if segment.size == 0 or not np.isfinite(segment).any() or np.nanmax(segment) <= 0:
            return self._midpoint(start, end)
        return start + int(np.nanargmax(segment))

    def _fill_points(self, start, end, anchors, count):
        """Fill remaining samples from the longest uncovered intervals."""

        if count <= 0:
            return []
        if end <= start:
            return [start] * count

        points = []
        occupied = sorted(set(int(x) for x in anchors if start <= int(x) < end))
        intervals = []
        cursor = start
        for anchor in occupied:
            intervals.append((cursor, anchor))
            cursor = anchor + 1
        intervals.append((cursor, end))
        intervals = [(right - left, left, right) for left, right in intervals if right > left]
        intervals.sort(reverse=True)
        for _, left, right in intervals:
            if len(points) >= count:
                break
            points.append(self._midpoint(left, right))

        candidate = start
        while len(points) < count:
            if candidate not in occupied and candidate not in points and candidate < end:
                points.append(candidate)
            candidate += 1
            if candidate >= end:
                break
        fallback = points[-1] if points else (occupied[-1] if occupied else self._midpoint(start, end))
        while len(points) < count:
            points.append(fallback)
        return points

    def _partition_points(self, start, end, velocity, acceleration):
        """Select three KNS-v1 frames from one coarse partition."""

        tv = self._segment_peak(velocity, start, end)
        ta = self._segment_peak(acceleration, start, end)
        if abs(tv - ta) <= self.merge_threshold:
            center = int(round((tv + ta) / 2))
            anchors = [min(max(center, start), max(start, end - 1))]
        else:
            anchors = [tv, ta]
        anchors = sorted(set(anchors))
        anchors.extend(self._fill_points(start, end, anchors, self.frames_per_partition - len(anchors)))
        return sorted(anchors[:self.frames_per_partition]), tv, ta

    def _get_kns_clip(self, keypoint, keypoint_score=None):
        """Build one KNS-v1 clip and return indices plus diagnostics."""

        total_frames = keypoint.shape[1]
        velocity, acceleration, confidence = self._motion_signals(keypoint, keypoint_score)
        num_partitions = self.clip_len // self.frames_per_partition
        bounds = np.linspace(0, total_frames, num_partitions + 1).astype(int)
        all_inds, peaks = [], []
        for idx in range(num_partitions):
            start = int(bounds[idx])
            end = int(bounds[idx + 1])
            if end <= start:
                end = min(total_frames, start + 1)
            points, tv, ta = self._partition_points(start, end, velocity, acceleration)
            all_inds.extend(points)
            peaks.append(dict(partition=idx, start=start, end=end, tv=int(tv), ta=int(ta)))
        inds = np.array(sorted(all_inds), dtype=np.int64)
        meta = dict(
            sampler='KNS-v1',
            velocity_peaks=[item['tv'] for item in peaks],
            acceleration_peaks=[item['ta'] for item in peaks],
            mean_confidence=float(np.mean(confidence)) if confidence.size else 0.0,
            global_video_length=int(total_frames),
            partition_peaks=peaks,
        )
        return inds, meta

    def __call__(self, results):
        if 'keypoint' not in results:
            return super().__call__(results)

        total_frames = results['total_frames']
        allinds, metas = [], []
        if self.uniform_clips:
            old_num_clips = self.num_clips
            self.num_clips = self.uniform_clips
            uniform = self._get_test_clips(total_frames, self.clip_len)
            self.num_clips = old_num_clips
            allinds.append(uniform)
            metas.extend(dict(sampler='uniform') for _ in range(self.uniform_clips))

        for _ in range(self.num_clips):
            inds, meta = self._get_kns_clip(results['keypoint'], results.get('keypoint_score'))
            allinds.append(inds)
            metas.append(meta)

        inds = np.concatenate(allinds)
        inds = np.mod(inds, total_frames)
        start_index = results['start_index']
        results['frame_inds'] = (inds + start_index).astype(int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.uniform_clips + self.num_clips
        results['kns_meta'] = metas
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'uniform_clips={self.uniform_clips}, '
                    f'frames_per_partition={self.frames_per_partition}, '
                    f'merge_threshold={self.merge_threshold}, '
                    f'seed={self.seed})')
        return repr_str


@PIPELINES.register_module()
class MotionAwareUniformSampleFrames(UniformSampleFrames):
    """Uniform sampler that biases training clips toward high-motion windows.

    This keeps the test-time behavior identical to ``UniformSampleFrames`` so
    score comparison is controlled, while training samples are drawn from the
    most dynamic temporal region when keypoints are available.

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to sample.
        p_interval (float | tuple[float, float]): Temporal crop ratio range.
        seed (int): Test-time random seed.
        motion_topk (int): Randomly choose among the top-k motion windows.
    """

    def __init__(self, clip_len, num_clips=1, p_interval=1, seed=255, motion_topk=3, **kwargs):
        super().__init__(clip_len, num_clips=num_clips, p_interval=p_interval, seed=seed, **kwargs)
        self.motion_topk = motion_topk

    @staticmethod
    def _motion_energy(keypoint):
        coords = keypoint[..., :2].astype(np.float32)
        valid = np.abs(coords).sum(axis=(-1, -2)) > 1e-5
        diff = np.abs(np.diff(coords, axis=1)).sum(axis=(-1, -2))
        valid_pair = valid[:, 1:] & valid[:, :-1]
        weighted = diff * valid_pair
        denom = np.maximum(valid_pair.sum(axis=0), 1)
        return weighted.sum(axis=0) / denom

    def _select_motion_window(self, keypoint, span):
        total_frames = keypoint.shape[1]
        if span >= total_frames:
            return 0, total_frames
        energy = self._motion_energy(keypoint)
        if energy.size == 0 or not np.isfinite(energy).any() or energy.max() <= 0:
            start = np.random.randint(total_frames - span + 1)
            return start, start + span
        if span <= 1:
            start = int(np.argmax(energy))
            return start, start + span
        padded = np.pad(energy, (1, 0), mode='constant')
        cumsum = np.cumsum(padded)
        # Motion has length T - 1. A frame window [s, s + span) owns motion
        # edges [s, s + span - 1).
        scores = cumsum[span - 1:] - cumsum[:-(span - 1)]
        if scores.size == 0:
            return 0, total_frames
        topk = min(self.motion_topk, scores.size)
        candidates = np.argsort(scores)[-topk:]
        start = int(np.random.choice(candidates))
        return start, start + span

    def _sample_from_window(self, start, end):
        span = end - start
        if span < self.clip_len:
            inner_start = np.random.randint(0, span)
            inds = np.arange(inner_start, inner_start + self.clip_len)
            return np.mod(inds, span) + start
        if self.clip_len <= span < 2 * self.clip_len:
            basic = np.arange(self.clip_len)
            extra = np.random.choice(self.clip_len + 1, span - self.clip_len, replace=False)
            offset = np.zeros(self.clip_len + 1, dtype=np.int64)
            offset[extra] = 1
            return basic + np.cumsum(offset)[:-1] + start
        bids = np.array([i * span // self.clip_len for i in range(self.clip_len + 1)])
        bsize = np.diff(bids)
        offset = np.random.randint(bsize)
        return bids[:self.clip_len] + offset + start

    def __call__(self, results):
        if results.get('test_mode', False) or 'keypoint' not in results:
            return super().__call__(results)

        total_frames = results['total_frames']
        allinds = []
        for _ in range(self.num_clips):
            ratio = np.random.rand() * (self.p_interval[1] - self.p_interval[0]) + self.p_interval[0]
            span = max(1, int(ratio * total_frames))
            start, end = self._select_motion_window(results['keypoint'], span)
            allinds.append(self._sample_from_window(start, end))

        inds = np.concatenate(allinds)
        inds = np.mod(inds, total_frames)
        results['frame_inds'] = (inds + results['start_index']).astype(int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results


@PIPELINES.register_module()
class UniformSampleDecode:

    def __init__(self, clip_len, num_clips=1, p_interval=1, seed=255):
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)

    # will directly return the decoded clips
    def _get_clips(self, full_kp, clip_len):
        M, T, V, C = full_kp.shape
        clips = []

        for clip_idx in range(self.num_clips):
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * T)
            off = np.random.randint(T - num_frames + 1)

            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = (np.arange(start, start + clip_len) % num_frames) + off
                clip = full_kp[:, inds].copy()
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                inds = basic + np.cumsum(offset)[:-1] + off
                clip = full_kp[:, inds].copy()
            else:
                bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset + off
                clip = full_kp[:, inds].copy()
            clips.append(clip)
        return np.concatenate(clips, 1)

    def _handle_dict(self, results):
        assert 'keypoint' in results
        kp = results.pop('keypoint')
        if 'keypoint_score' in results:
            kp_score = results.pop('keypoint_score')
            kp = np.concatenate([kp, kp_score[..., None]], axis=-1)

        kp = kp.astype(np.float32)
        # start_index will not be used
        kp = self._get_clips(kp, self.clip_len)

        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        results['keypoint'] = kp
        return results

    def _handle_list(self, results):
        assert len(results) == self.num_clips
        self.num_clips = 1
        clips = []
        for res in results:
            assert 'keypoint' in res
            kp = res.pop('keypoint')
            if 'keypoint_score' in res:
                kp_score = res.pop('keypoint_score')
                kp = np.concatenate([kp, kp_score[..., None]], axis=-1)

            kp = kp.astype(np.float32)
            kp = self._get_clips(kp, self.clip_len)
            clips.append(kp)
        ret = cp.deepcopy(results[0])
        ret['clip_len'] = self.clip_len
        ret['frame_interval'] = None
        ret['num_clips'] = len(results)
        ret['keypoint'] = np.concatenate(clips, 1)
        self.num_clips = len(results)
        return ret

    def __call__(self, results):
        test_mode = results.get('test_mode', False)
        if test_mode is True:
            np.random.seed(self.seed)
        if isinstance(results, list):
            return self._handle_list(results)
        else:
            return self._handle_dict(results)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'p_interval={self.p_interval}, '
                    f'seed={self.seed})')
        return repr_str


@PIPELINES.register_module()
class SampleFrames:
    """Sample frames from the video.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 start_index=None,
                 keep_tail_frames=False,
                 **deprecated_kwargs):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.keep_tail_frames = keep_tail_frames
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warning_r0('No longer support "start_index" in "SampleFrames", '
                       'it should be set in dataset class, see this pr: '
                       'https://github.com/open-mmlab/mmaction2/pull/89')
        if len(deprecated_kwargs):
            warning_r0('[UniformSampleFrames] The following args has been deprecated: ')
            for k, v in deprecated_kwargs.items():
                warning_r0(f'Arg name: {k}; Arg value: {v}')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(int)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=int)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=int)
        return clip_offsets

    def _sample_clips(self, num_frames, test_mode=False):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        clip_offsets = self._sample_clips(total_frames, results.get('test_mode', False))
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt})')
        return repr_str
